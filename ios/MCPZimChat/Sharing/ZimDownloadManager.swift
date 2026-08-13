// SPDX-License-Identifier: MIT

import Foundation
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// In-app downloader for catalog archives (Wikipedia / StreetZIM ZIMs).
///
/// Runs on a **background `URLSession`**, so on iOS a multi-gigabyte download
/// keeps transferring after the user leaves the app or the screen locks, and
/// survives app termination — the system relaunches the app to hand back the
/// finished file (see `handleEventsForBackgroundURLSession` in the app
/// delegate). While the app *is* frontmost with downloads running,
/// `SleepBlocker` additionally keeps the device awake so a plugged-in phone
/// on a shelf finishes overnight without ever suspending.
///
/// Finished files are moved straight into the app's Documents folder, which
/// is the ZIM library's scan root — a completed download is a loaded library
/// entry one `addReaders` call (or one launch-time scan) later. Partial
/// downloads live in the session's own temp storage and never pollute the
/// library.
@MainActor
final class ZimDownloadManager: NSObject, ObservableObject {
    static let shared = ZimDownloadManager()
    static let sessionIdentifier = "com.tiltastech.zimfo.zim-downloads"

    /// Posted (on the main queue) when a finished archive has landed in
    /// Documents, with `["url": URL]` — the foreground fast-path for the
    /// library to open it immediately. Background completions are picked up
    /// by the next launch's Documents scan instead.
    static let fileReadyNotification = Notification.Name("ZimDownloadManager.fileReady")

    enum ItemState: Equatable {
        case downloading
        case paused
        case failed(String)
        case finished
    }

    struct Item: Identifiable, Equatable {
        let id: String            // stable catalog id (survives file-date bumps)
        let title: String
        let url: URL
        var expectedBytes: Int64  // catalog estimate until the server says better
        var receivedBytes: Int64 = 0
        var bytesPerSecond: Double = 0
        var state: ItemState = .downloading
        var destination: URL? = nil

        var filename: String { url.lastPathComponent }
        var fractionComplete: Double {
            guard state != .finished else { return 1 }
            guard expectedBytes > 0 else { return 0 }
            return min(1, Double(receivedBytes) / Double(expectedBytes))
        }
    }

    @Published private(set) var items: [Item] = []

    /// Stored by the app delegate when iOS relaunches us for background
    /// session events; called once the session says it's drained.
    var backgroundEventsCompletionHandler: (() -> Void)?

    private var taskIDToItemID: [Int: String] = [:]
    private var rateClock: [String: (time: TimeInterval, bytes: Int64)] = [:]
    private let shim = SessionDelegateShim()

    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.background(withIdentifier: Self.sessionIdentifier)
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.waitsForConnectivity = true
        // Multi-day ceiling: a 49 GB archive on hotel Wi-Fi is a legitimate
        // multi-night download, and the resource timer keeps running while
        // the app is suspended.
        config.timeoutIntervalForResource = 7 * 24 * 3600
        return URLSession(configuration: config, delegate: shim, delegateQueue: nil)
    }()

    private override init() {
        super.init()
        shim.manager = self
        // Reconnect to whatever the background session was doing before this
        // launch (downloads keep running between launches).
        session.getAllTasks { [weak self] tasks in
            let snapshots = tasks.compactMap { task -> (TaskLabel, Int, Int64, Int64, Bool)? in
                guard let label = TaskLabel(taskDescription: task.taskDescription) else { return nil }
                return (label, task.taskIdentifier,
                        task.countOfBytesReceived, task.countOfBytesExpectedToReceive,
                        task.state == .running || task.state == .suspended)
            }
            Task { @MainActor in
                self?.adoptRestoredTasks(snapshots)
            }
        }
    }

    // MARK: - Public controls

    /// True only for rows that are *both* marked `.downloading` and still own
    /// a task. The task-ownership half matters: a row whose task went away
    /// while the state said "downloading" used to pin `isIdleTimerDisabled`
    /// for the rest of the process (battery drain with nothing transferring —
    /// review 2026-08-13, "Fix first" #5). Every mutator recomputes the
    /// keep-awake flag from this, so no path can leave it stuck on.
    var hasActiveDownloads: Bool {
        let owningTask = Set(taskIDToItemID.values)
        return items.contains { $0.state == .downloading && owningTask.contains($0.id) }
    }

    /// True while the item is downloading or paused (i.e. occupying the list).
    func isInFlight(id: String) -> Bool {
        guard let item = items.first(where: { $0.id == id }) else { return false }
        switch item.state {
        case .downloading, .paused: return true
        case .failed, .finished: return false
        }
    }

    /// True if an archive with this catalog item's filename is already in the
    /// library folder (from this downloader, a browser hand-off, or a friend).
    nonisolated static func alreadyInLibrary(filename: String) -> Bool {
        guard let docs = try? FileManager.default.url(for: .documentDirectory,
                                                      in: .userDomainMask,
                                                      appropriateFor: nil,
                                                      create: false) else { return false }
        return FileManager.default.fileExists(atPath: docs.appendingPathComponent(filename).path)
    }

    /// Free bytes on the library volume, counting purgeable space the system
    /// can reclaim for "important" writes.
    nonisolated static func availableLibraryBytes() -> Int64? {
        guard let docs = try? FileManager.default.url(for: .documentDirectory,
                                                      in: .userDomainMask,
                                                      appropriateFor: nil,
                                                      create: false) else { return nil }
        let values = try? docs.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    func download(_ catalogItem: ZimCatalogItem) {
        // `defer` rather than a trailing call so the early returns below (and
        // in every sibling mutator) still recompute the keep-awake flag.
        defer { updateSleepBlocker() }
        // Re-tapping an in-flight item is a no-op; a failed/finished row is
        // replaced by the fresh attempt.
        if isInFlight(id: catalogItem.id) { return }
        items.removeAll { $0.id == catalogItem.id }

        var item = Item(id: catalogItem.id,
                        title: catalogItem.title,
                        url: catalogItem.url,
                        expectedBytes: catalogItem.sizeBytes)
        item.state = .downloading
        items.append(item)

        let label = TaskLabel(id: catalogItem.id, title: catalogItem.title,
                              urlString: catalogItem.url.absoluteString,
                              expectedBytes: catalogItem.sizeBytes)
        let task = session.downloadTask(with: catalogItem.url)
        task.taskDescription = label.encoded
        taskIDToItemID[task.taskIdentifier] = catalogItem.id
        task.resume()
    }

    func pause(id: String) {
        defer { updateSleepBlocker() }
        guard let index = items.firstIndex(where: { $0.id == id }),
              items[index].state == .downloading else { return }
        items[index].state = .paused
        items[index].bytesPerSecond = 0
        let item = items[index]
        let label = TaskLabel(id: item.id, title: item.title,
                              urlString: item.url.absoluteString,
                              expectedBytes: item.expectedBytes)
        let taskIDs = taskIDToItemID.filter { $0.value == id }.map(\.key)
        // Hand ownership back in the same main-actor step that sets `.paused`,
        // so state and task ownership can never disagree while the async
        // `getAllTasks` cancel is in flight.
        taskIDToItemID = taskIDToItemID.filter { $0.value != id }
        rateClock[id] = nil
        session.getAllTasks { tasks in
            for task in tasks where taskIDs.contains(task.taskIdentifier) {
                (task as? URLSessionDownloadTask)?.cancel { resumeData in
                    Self.writeResumeData(resumeData, id: id)
                    Self.writeResumeLabel(label, id: id)
                }
            }
        }
    }

    func resume(id: String) {
        defer { updateSleepBlocker() }
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        switch items[index].state {
        case .downloading, .finished: return
        case .paused, .failed: break
        }
        let item = items[index]
        items[index].state = .downloading
        let label = TaskLabel(id: item.id, title: item.title,
                              urlString: item.url.absoluteString,
                              expectedBytes: item.expectedBytes)
        let task: URLSessionDownloadTask
        if let resumeData = Self.readResumeData(id: id) {
            task = session.downloadTask(withResumeData: resumeData)
        } else {
            task = session.downloadTask(with: item.url)
        }
        Self.deleteResumeData(id: id)
        task.taskDescription = label.encoded
        taskIDToItemID[task.taskIdentifier] = id
        task.resume()
    }

    /// Cancels a transfer and forgets its partial data. Finished rows are
    /// simply dismissed (the file in the library is kept).
    func cancel(id: String) {
        defer { updateSleepBlocker() }
        let taskIDs = taskIDToItemID.filter { $0.value == id }.map(\.key)
        session.getAllTasks { tasks in
            for task in tasks where taskIDs.contains(task.taskIdentifier) {
                task.cancel()
            }
        }
        Self.deleteResumeData(id: id)
        items.removeAll { $0.id == id }
        taskIDToItemID = taskIDToItemID.filter { $0.value != id }
        rateClock[id] = nil
    }

    // MARK: - Delegate plumbing (called from the session's queue via the shim)

    fileprivate func adoptRestoredTasks(_ snapshots: [(label: TaskLabel, taskID: Int,
                                                      received: Int64, expected: Int64,
                                                      live: Bool)]) {
        defer { updateSleepBlocker() }
        for snapshot in snapshots where snapshot.live {
            guard !items.contains(where: { $0.id == snapshot.label.id }),
                  let url = URL(string: snapshot.label.urlString) else { continue }
            var item = Item(id: snapshot.label.id, title: snapshot.label.title,
                            url: url, expectedBytes: snapshot.label.expectedBytes)
            item.receivedBytes = max(0, snapshot.received)
            if snapshot.expected > 0 { item.expectedBytes = snapshot.expected }
            items.append(item)
            taskIDToItemID[snapshot.taskID] = snapshot.label.id
        }
        // A previous run may also have failed tasks whose resume blobs are on
        // disk but which have no live task anymore — resurface them as paused
        // so they're resumable rather than silently gone.
        for id in Self.storedResumeIDs() where !items.contains(where: { $0.id == id }) {
            guard let label = Self.readResumeLabel(id: id),
                  let url = URL(string: label.urlString) else { continue }
            var item = Item(id: label.id, title: label.title, url: url,
                            expectedBytes: label.expectedBytes)
            item.state = .paused
            items.append(item)
        }
    }

    fileprivate func progress(taskID: Int, label: TaskLabel?,
                              received: Int64, expected: Int64) {
        defer { updateSleepBlocker() }
        guard let id = itemID(taskID: taskID, label: label),
              let index = items.firstIndex(where: { $0.id == id }) else { return }
        // A `didWriteData` callback already in flight when the user paused
        // lands here afterwards. Promoting it back to `.downloading` wedged
        // the row: the cancel then completes as `NSURLErrorCancelled` (which
        // the shim drops), leaving "downloading" with no task — `resume()`
        // early-returns and the keep-awake flag never clears (review
        // 2026-08-13, "Fix first" #5). A paused row owns no task, so trailing
        // writes from the one we cancelled are ignored entirely.
        guard items[index].state != .paused else {
            // `itemID` re-adopts an unmapped task from its label; undo that
            // for the task we just cancelled so ownership stays accurate.
            taskIDToItemID[taskID] = nil
            return
        }
        items[index].receivedBytes = received
        if expected > 0 { items[index].expectedBytes = expected }
        if items[index].state != .downloading { items[index].state = .downloading }

        let now = ProcessInfo.processInfo.systemUptime
        if let last = rateClock[id] {
            let dt = now - last.time
            if dt >= 0.5 {
                let instant = Double(received - last.bytes) / dt
                let previous = items[index].bytesPerSecond
                items[index].bytesPerSecond = previous == 0 ? instant : previous * 0.7 + instant * 0.3
                rateClock[id] = (now, received)
            }
        } else {
            rateClock[id] = (now, received)
        }
    }

    fileprivate func finished(taskID: Int, label: TaskLabel?, destination: URL) {
        defer { updateSleepBlocker() }
        guard let id = itemID(taskID: taskID, label: label) else { return }
        Self.deleteResumeData(id: id)
        if let index = items.firstIndex(where: { $0.id == id }) {
            items[index].state = .finished
            items[index].destination = destination
            items[index].bytesPerSecond = 0
            items[index].receivedBytes = max(items[index].receivedBytes, items[index].expectedBytes)
        }
        rateClock[id] = nil
        NotificationCenter.default.post(name: Self.fileReadyNotification,
                                        object: nil,
                                        userInfo: ["url": destination])
    }

    fileprivate func failed(taskID: Int, label: TaskLabel?, message: String, hasResumeData: Bool) {
        defer { updateSleepBlocker() }
        guard let id = itemID(taskID: taskID, label: label) else { return }
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        // A user-initiated pause also lands here (cancel error) — keep the
        // paused state it already has.
        if items[index].state == .paused { return }
        items[index].state = hasResumeData ? .paused : .failed(message)
        items[index].bytesPerSecond = 0
        rateClock[id] = nil
        // The row no longer owns a task; drop the mapping so a paused/failed
        // item can't be counted as active by `hasActiveDownloads`.
        taskIDToItemID[taskID] = nil
    }

    fileprivate func backgroundEventsDrained() {
        backgroundEventsCompletionHandler?()
        backgroundEventsCompletionHandler = nil
    }

    private func itemID(taskID: Int, label: TaskLabel?) -> String? {
        if let id = taskIDToItemID[taskID] { return id }
        guard let label else { return nil }
        // Task from a previous process (background relaunch): adopt it now.
        taskIDToItemID[taskID] = label.id
        if !items.contains(where: { $0.id == label.id }), let url = URL(string: label.urlString) {
            items.append(Item(id: label.id, title: label.title, url: url,
                              expectedBytes: label.expectedBytes))
        }
        return label.id
    }

    private func updateSleepBlocker() {
        SleepBlocker.set(hasActiveDownloads, reason: "zim-downloads")
    }

    // MARK: - Resume-data persistence (survives relaunches)

    private nonisolated static func resumeDirectory() -> URL? {
        guard let base = try? FileManager.default.url(for: .applicationSupportDirectory,
                                                      in: .userDomainMask,
                                                      appropriateFor: nil, create: true)
        else { return nil }
        let dir = base.appendingPathComponent("ZimDownloads", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private nonisolated static func safeName(_ id: String) -> String {
        id.replacingOccurrences(of: "[^A-Za-z0-9._-]", with: "_", options: .regularExpression)
    }

    fileprivate nonisolated static func writeResumeData(_ data: Data?, id: String) {
        guard let data, let dir = resumeDirectory() else { return }
        try? data.write(to: dir.appendingPathComponent(safeName(id) + ".resume"))
    }

    private nonisolated static func readResumeData(id: String) -> Data? {
        guard let dir = resumeDirectory() else { return nil }
        return try? Data(contentsOf: dir.appendingPathComponent(safeName(id) + ".resume"))
    }

    fileprivate nonisolated static func deleteResumeData(id: String) {
        guard let dir = resumeDirectory() else { return }
        try? FileManager.default.removeItem(at: dir.appendingPathComponent(safeName(id) + ".resume"))
        try? FileManager.default.removeItem(at: dir.appendingPathComponent(safeName(id) + ".label"))
    }

    fileprivate nonisolated static func writeResumeLabel(_ label: TaskLabel, id: String) {
        guard let dir = resumeDirectory(), let data = label.encoded.data(using: .utf8) else { return }
        try? data.write(to: dir.appendingPathComponent(safeName(id) + ".label"))
    }

    private nonisolated static func readResumeLabel(id: String) -> TaskLabel? {
        guard let dir = resumeDirectory(),
              let data = try? Data(contentsOf: dir.appendingPathComponent(safeName(id) + ".label")),
              let text = String(data: data, encoding: .utf8) else { return nil }
        return TaskLabel(taskDescription: text)
    }

    private nonisolated static func storedResumeIDs() -> [String] {
        guard let dir = resumeDirectory(),
              let files = try? FileManager.default.contentsOfDirectory(at: dir,
                                                                       includingPropertiesForKeys: nil)
        else { return [] }
        return files.filter { $0.pathExtension == "resume" }
            .map { $0.deletingPathExtension().lastPathComponent }
    }
}

// MARK: - Task labeling

/// The identity we stamp onto each `URLSessionTask.taskDescription`, so a
/// relaunched process (background session hand-back) can reconstruct which
/// catalog item a task belongs to without any other persisted registry.
struct TaskLabel {
    let id: String
    let title: String
    let urlString: String
    let expectedBytes: Int64

    var encoded: String {
        let payload: [String: Any] = ["id": id, "title": title,
                                      "url": urlString, "bytes": expectedBytes]
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let text = String(data: data, encoding: .utf8) else { return id }
        return text
    }

    init(id: String, title: String, urlString: String, expectedBytes: Int64) {
        self.id = id
        self.title = title
        self.urlString = urlString
        self.expectedBytes = expectedBytes
    }

    init?(taskDescription: String?) {
        guard let taskDescription,
              let data = taskDescription.data(using: .utf8),
              let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let id = payload["id"] as? String,
              let title = payload["title"] as? String,
              let url = payload["url"] as? String
        else { return nil }
        self.id = id
        self.title = title
        self.urlString = url
        self.expectedBytes = (payload["bytes"] as? NSNumber)?.int64Value ?? 0
    }
}

// MARK: - URLSession delegate shim

/// Nonisolated delegate target. `didFinishDownloadingTo` must move the temp
/// file *before returning* (the system deletes it afterward), so the move
/// happens here on the session queue; everything else hops to the manager on
/// the main actor.
private final class SessionDelegateShim: NSObject, URLSessionDownloadDelegate {
    weak var manager: ZimDownloadManager?

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        let label = TaskLabel(taskDescription: downloadTask.taskDescription)
        let taskID = downloadTask.taskIdentifier
        Task { @MainActor [weak manager] in
            manager?.progress(taskID: taskID, label: label,
                              received: totalBytesWritten,
                              expected: totalBytesExpectedToWrite)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        let label = TaskLabel(taskDescription: downloadTask.taskDescription)
        let taskID = downloadTask.taskIdentifier
        let filename = downloadTask.originalRequest?.url?.lastPathComponent
            ?? URL(string: label?.urlString ?? "")?.lastPathComponent
            ?? location.lastPathComponent
        let fm = FileManager.default
        do {
            guard let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask,
                                         appropriateFor: nil, create: true) else {
                throw CocoaError(.fileNoSuchFile)
            }
            let destination = docs.appendingPathComponent(filename)
            if fm.fileExists(atPath: destination.path) {
                // Same filename == same published archive; replace the copy
                // (covers a re-download after a corrupted first attempt).
                try fm.removeItem(at: destination)
            }
            try fm.moveItem(at: location, to: destination)
            Task { @MainActor [weak manager] in
                manager?.finished(taskID: taskID, label: label, destination: destination)
            }
        } catch {
            let message = "Couldn't save \(filename): \(error.localizedDescription)"
            Task { @MainActor [weak manager] in
                manager?.failed(taskID: taskID, label: label, message: message, hasResumeData: false)
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        guard let error else { return } // success already handled above
        let nsError = error as NSError
        if nsError.code == NSURLErrorCancelled { return } // pause/cancel path
        let label = TaskLabel(taskDescription: task.taskDescription)
        let taskID = task.taskIdentifier
        let resumeData = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        if let resumeData, let label {
            ZimDownloadManager.writeResumeData(resumeData, id: label.id)
            ZimDownloadManager.writeResumeLabel(label, id: label.id)
        }
        let message = nsError.localizedDescription
        Task { @MainActor [weak manager] in
            manager?.failed(taskID: taskID, label: label, message: message,
                            hasResumeData: resumeData != nil)
        }
    }

    #if os(iOS)
    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor [weak manager] in
            manager?.backgroundEventsDrained()
        }
    }
    #endif
}

// MARK: - Keep-awake while transfers run

/// Ref-counted "don't sleep" switch shared by the HTTP downloader and the
/// nearby-share engine. iOS: disables the idle timer (screen may dim but the
/// device never suspends the foreground app mid-transfer). macOS: takes a
/// process-level "idle system sleep" assertion.
@MainActor
enum SleepBlocker {
    private static var reasons: Set<String> = []
    #if os(macOS)
    private static var activityToken: NSObjectProtocol?
    #endif

    static func set(_ blocked: Bool, reason: String) {
        if blocked { reasons.insert(reason) } else { reasons.remove(reason) }
        apply()
    }

    private static func apply() {
        let active = !reasons.isEmpty
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = active
        #elseif os(macOS)
        if active, activityToken == nil {
            activityToken = ProcessInfo.processInfo.beginActivity(
                options: [.idleSystemSleepDisabled, .userInitiated],
                reason: "Transferring offline archives")
        } else if !active, let token = activityToken {
            ProcessInfo.processInfo.endActivity(token)
            activityToken = nil
        }
        #endif
    }
}
