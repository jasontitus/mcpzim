// SPDX-License-Identifier: MIT

import Foundation
import CryptoKit
import MCPZimKit
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
    static let shared = ZimDownloadManager(restoringDownloads: true)
    static let sessionIdentifier = "com.tiltastech.zimfo.zim-downloads"

    /// Posted (on the main queue) when a finished archive has landed in
    /// Documents, with `["url": URL]` — the foreground fast-path for the
    /// library to open it immediately. Background completions are picked up
    /// by the next launch's Documents scan instead.
    static let fileReadyNotification = Notification.Name("ZimDownloadManager.fileReady")

    /// Posted when a model download has landed in its cache slot, with
    /// `["url": URL, "id": String]` — the foreground fast-path for the
    /// provider to resume its awaiting load. Background completions are
    /// picked up by the next launch's cache-hit check.
    static let modelReadyNotification = Notification.Name("ZimDownloadManager.modelReady")

    /// Headroom (bytes) reserved above a model's advertised size when the
    /// storage gate decides whether a download can start. Download staging
    /// and the final rename both live on the same volume, so the margin only
    /// covers transient file-system slack, not a second copy.
    static let modelSafetyMargin: Int64 = 256 * 1024 * 1024

    enum ItemState: Equatable {
        case downloading
        case waitingForNetwork
        case paused
        case failed(String)
        case finished
    }

    /// What a row's finished file feeds. ZIM archives go into the library's
    /// Documents scan root; model weights go into the model cache slot.
    enum TransferKind: String {
        case zim
        case model
    }

    struct Item: Identifiable, Equatable {
        let id: String            // stable catalog id (survives file-date bumps)
        let title: String
        let url: URL
        var kind: TransferKind = .zim
        var expectedBytes: Int64  // catalog estimate until the server says better
        var receivedBytes: Int64 = 0
        var bytesPerSecond: Double = 0
        var state: ItemState = .downloading
        var destination: URL? = nil
        /// Model rows only: expected SHA-256 of the finished weights. Pinned
        /// so a truncated or corrupt GGUF can never be handed to llama.cpp.
        var sha256: String? = nil

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
    // IDs never repeat within a URLSession. Keep tombstones for the lifetime
    // of this process so callbacks already queued at cancellation cannot adopt.
    private var retiredTaskIDs: Set<Int> = []
    private var pendingPause: [String: UUID] = [:]
    private var resumeAfterPause: Set<String> = []
    private var rateClock: [String: (time: TimeInterval, bytes: Int64)] = [:]
    /// Awaited by a `LlamaCppProvider` that handed its GGUF download to us.
    /// Resumed on the main actor from `finished`/`failed` (foreground) or
    /// dropped when the download completes in the background (the next
    /// launch's cache-hit check picks the file up instead).
    private var modelCompletions: [String: CheckedContinuation<URL, Error>] = [:]
    /// Per-model progress bridge into the provider's `.downloading(fraction)`
    /// state stream. Keyed by item id, cleared on finish/fail.
    private var modelProgressCallbacks: [String: @MainActor (Double) -> Void] = [:]
    /// Per-model "the OS parked the task waiting for connectivity" signal,
    /// so the provider can surface `.waitingForNetwork` instead of a frozen
    /// 0% and the launch setup won't mistake it for a stall.
    private var modelWaitingCallbacks: [String: @MainActor (Bool) -> Void] = [:]
    private let shim = SessionDelegateShim()
    private let restoresDownloads: Bool

    private lazy var session: URLSession = {
        let config = restoresDownloads
            ? URLSessionConfiguration.background(withIdentifier: Self.sessionIdentifier)
            : URLSessionConfiguration.ephemeral
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.waitsForConnectivity = true
        // Multi-day ceiling: a 49 GB archive on hotel Wi-Fi is a legitimate
        // multi-night download, and the resource timer keeps running while
        // the app is suspended.
        config.timeoutIntervalForResource = 7 * 24 * 3600
        return URLSession(configuration: config, delegate: shim, delegateQueue: nil)
    }()

    init(restoringDownloads: Bool) {
        restoresDownloads = restoringDownloads
        super.init()
        shim.manager = self
        guard restoringDownloads else { return }
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
        case .downloading, .waitingForNetwork, .paused: return true
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

    /// Free bytes on the volume that holds `url` (purgeable-inclusive). Used
    /// by the model storage gate so a multi-GB GGUF never starts on a disk
    /// that can't hold it. Walks up to an existing ancestor — the model cache
    /// slot's directory may not exist yet on a fresh install.
    nonisolated static func availableVolumeBytes(for url: URL) -> Int64? {
        var cursor = url.deletingLastPathComponent()
        while !FileManager.default.fileExists(atPath: cursor.path), cursor.path != "/" {
            cursor.deleteLastPathComponent()
        }
        let values = try? cursor.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    /// Lowercase hex SHA-256 of a file, streamed so a multi-GB GGUF never
    /// loads into memory. Throws on read error.
    nonisolated static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        // FileHandle can return autoreleased NSData backing each Data. A
        // detached task/session delegate need not drain its pool until this
        // entire function returns: without this inner pool a 3.8 GB checksum
        // can retain 3.8 GB of read buffers and kill the app before model load.
        while try autoreleasepool(invoking: { () throws -> Bool in
            guard let chunk = try handle.read(upToCount: 1 << 20),
                  !chunk.isEmpty else { return false }
            hasher.update(data: chunk)
            return true
        }) {}
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    nonisolated static func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }

    func download(_ catalogItem: ZimCatalogItem) {
        startTransfer(id: catalogItem.id, title: catalogItem.title,
                      url: catalogItem.url, expectedBytes: catalogItem.sizeBytes,
                      kind: .zim, sha256: nil, destination: nil)
    }

    /// Start (or re-arm) a background transfer for a model's weights.
    ///
    /// Reuses the exact background `URLSession` machinery the ZIM catalog
    /// uses, so a multi-gigabyte GGUF keeps transferring after the user
    /// leaves or the screen locks, survives app termination (the system
    /// relaunches us to hand the file back), and can resume a partial pull
    /// across relaunches. The provider awaits the returned URL; if the
    /// download finishes while the app isn't running, the file simply lands
    /// in its cache slot and the next launch's cache-hit check finds it.
    ///
    /// Safety gates before and during:
    ///   * storage is checked first — a multi-GB pull never starts on a
    ///     volume that can't hold it, and the error names the shortfall;
    ///   * the transfer writes to a staged `.partial` file and is only moved
    ///     into the cache slot after the expected size **and** SHA-256
    ///     (when pinned) validate — a truncated/corrupt GGUF can never
    ///     replace a working model;
    ///   * a paused/failed model row is resumed instead of duplicated, and an
    ///     in-flight one is awaited rather than restarted.
    func downloadModel(
        spec: ModelDownloadSpec,
        onProgress: @escaping @MainActor (Double) -> Void,
        onWaiting: @escaping @MainActor (Bool) -> Void = { _ in }
    ) async throws -> URL {
        LogArchive.shared.trace("downloadModel: \(spec.title) · bytes=\(spec.expectedBytes)")
        // Storage pre-check: explain the requirement before touching the
        // network, and never orphan a half-written file on a full disk.
        if spec.expectedBytes > 0 {
            let required = spec.expectedBytes + Self.modelSafetyMargin
            if let free = Self.availableVolumeBytes(for: spec.destination),
               required > free {
                throw ZimDownloadError.insufficientStorage(
                    needed: required, available: free, title: spec.title)
            }
        }

        // Reuse an existing row for this model id instead of stacking
        // duplicates (re-selecting a model while it's mid-download, or
        // reloading after a pause, must not start a second transfer).
        if let index = items.firstIndex(where: { $0.id == spec.id }) {
            switch items[index].state {
            case .finished:
                if let dest = items[index].destination { return dest }
            case .paused, .failed:
                resume(id: spec.id)
            case .downloading, .waitingForNetwork:
                break
            }
        }
        // Only start a fresh transfer when nothing in-flight remains for this
        // id (an empty list, or a finished row that somehow lost its
        // destination). A paused/failed row was just resumed above.
        if !isInFlight(id: spec.id) {
            startTransfer(id: spec.id, title: spec.title, url: spec.url,
                          expectedBytes: spec.expectedBytes, kind: .model,
                          sha256: spec.sha256, destination: spec.destination)
        }

        modelProgressCallbacks[spec.id] = onProgress
        modelWaitingCallbacks[spec.id] = onWaiting
        return try await withCheckedThrowingContinuation { cont in
            modelCompletions[spec.id] = cont
        }
    }

    /// Shared transfer kickoff for both ZIM archives and model weights.
    /// Re-tapping an in-flight item is a no-op; a failed/finished row is
    /// replaced by the fresh attempt.
    private func startTransfer(id: String, title: String, url: URL,
                               expectedBytes: Int64, kind: TransferKind,
                               sha256: String?, destination: URL?) {
        // `defer` rather than a trailing call so the early returns below (and
        // in every sibling mutator) still recompute the keep-awake flag.
        defer { updateSleepBlocker() }
        if isInFlight(id: id) { return }
        items.removeAll { $0.id == id }

        var item = Item(id: id, title: title, url: url,
                        expectedBytes: expectedBytes)
        item.kind = kind
        item.sha256 = sha256
        item.destination = destination
        item.state = .downloading
        items.append(item)

        let label = TaskLabel(id: id, title: title, urlString: url.absoluteString,
                              expectedBytes: expectedBytes, kind: kind.rawValue,
                              sha256: sha256, destPath: destination?.path)
        let task = session.downloadTask(with: url)
        task.taskDescription = label.encoded
        taskIDToItemID[task.taskIdentifier] = id
        task.resume()
    }

    func pause(id: String) {
        defer { updateSleepBlocker() }
        guard let index = items.firstIndex(where: { $0.id == id }),
              items[index].state == .downloading || items[index].state == .waitingForNetwork
        else { return }
        items[index].state = .paused
        items[index].bytesPerSecond = 0
        let item = items[index]
        let label = TaskLabel(id: item.id, title: item.title,
                              urlString: item.url.absoluteString,
                              expectedBytes: item.expectedBytes,
                              kind: item.kind.rawValue,
                              sha256: item.sha256,
                              destPath: item.destination?.path)
        let taskIDs = taskIDToItemID.filter { $0.value == id }.map(\.key)
        // Hand ownership back in the same main-actor step that sets `.paused`,
        // so state and task ownership can never disagree while the async
        // `getAllTasks` cancel is in flight.
        retiredTaskIDs.formUnion(taskIDs)
        taskIDToItemID = taskIDToItemID.filter { $0.value != id }
        rateClock[id] = nil
        let pauseID = UUID()
        pendingPause[id] = pauseID
        session.getAllTasks { tasks in
            let owned = tasks.compactMap { task -> URLSessionDownloadTask? in
                taskIDs.contains(task.taskIdentifier) ? task as? URLSessionDownloadTask : nil
            }
            if owned.isEmpty {
                Task { @MainActor [weak self] in
                    self?.pauseCompleted(id: id, pauseID: pauseID, label: label, resumeData: nil)
                }
            }
            for task in owned {
                task.cancel { resumeData in
                    Task { @MainActor [weak self] in
                        self?.pauseCompleted(id: id, pauseID: pauseID, label: label, resumeData: resumeData)
                    }
                }
            }
        }
    }

    private func pauseCompleted(id: String, pauseID: UUID, label: TaskLabel, resumeData: Data?) {
        guard pendingPause[id] == pauseID else { return }
        pendingPause[id] = nil
        guard items.contains(where: { $0.id == id && $0.state == .paused }) else { return }
        Self.writeResumeData(resumeData, id: id)
        Self.writeResumeLabel(label, id: id)
        if resumeAfterPause.remove(id) != nil { resume(id: id) }
    }

    func resume(id: String) {
        defer { updateSleepBlocker() }
        if pendingPause[id] != nil { resumeAfterPause.insert(id); return }
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        switch items[index].state {
        case .downloading, .finished: return
        case .waitingForNetwork:
            // The task is alive but parked on connectivity. "Retry" means
            // tear it down and start a fresh attempt so it re-probes the
            // network instead of sitting suspended.
            cancelCurrentTask(id: id)
        case .paused, .failed: break
        }
        let item = items[index]
        items[index].state = .downloading
        let label = TaskLabel(id: item.id, title: item.title,
                              urlString: item.url.absoluteString,
                              expectedBytes: item.expectedBytes,
                              kind: item.kind.rawValue,
                              sha256: item.sha256,
                              destPath: item.destination?.path)
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

    /// Cancel the task(s) owned by an item without dropping the row. Used by
    /// `resume` when retrying a `.waitingForNetwork` row — the live task is
    /// parked on connectivity and would otherwise never re-probe.
    private func cancelCurrentTask(id: String) {
        let taskIDs = taskIDToItemID.filter { $0.value == id }.map(\.key)
        retiredTaskIDs.formUnion(taskIDs)
        taskIDToItemID = taskIDToItemID.filter { $0.value != id }
        rateClock[id] = nil
        session.getAllTasks { tasks in
            for task in tasks where taskIDs.contains(task.taskIdentifier) {
                task.cancel()
            }
        }
    }

    /// Cancels a transfer and forgets its partial data. Finished rows are
    /// simply dismissed (the file in the library is kept).
    func cancel(id: String) {
        defer { updateSleepBlocker() }
        pendingPause[id] = nil
        resumeAfterPause.remove(id)
        let taskIDs = taskIDToItemID.filter { $0.value == id }.map(\.key)
        retiredTaskIDs.formUnion(taskIDs)
        session.getAllTasks { tasks in
            for task in tasks where taskIDs.contains(task.taskIdentifier) {
                task.cancel()
            }
        }
        Self.deleteResumeData(id: id)
        modelProgressCallbacks[id] = nil
        modelWaitingCallbacks[id] = nil
        // A model row being cancelled while the provider awaits it must fail
        // the load — otherwise the continuation leaks and `load()` hangs.
        if let cont = modelCompletions.removeValue(forKey: id) {
            cont.resume(throwing: CancellationError())
        }
        items.removeAll { $0.id == id }
        taskIDToItemID = taskIDToItemID.filter { $0.value != id }
        rateClock[id] = nil
    }

    // MARK: - Delegate plumbing (called from the session's queue via the shim)

    func adoptRestoredTasks(_ snapshots: [(label: TaskLabel, taskID: Int,
                                                      received: Int64, expected: Int64,
                                                      live: Bool)]) {
        defer { updateSleepBlocker() }
        for snapshot in snapshots where snapshot.live && !retiredTaskIDs.contains(snapshot.taskID) {
            guard !items.contains(where: { $0.id == snapshot.label.id }),
                  let url = URL(string: snapshot.label.urlString) else { continue }
            var item = Self.makeItem(label: snapshot.label, url: url)
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
            var item = Self.makeItem(label: label, url: url)
            item.state = .paused
            items.append(item)
        }
    }

    func progress(taskID: Int, label: TaskLabel?,
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
                let instant = Double(max(0, received - last.bytes)) / dt
                let previous = items[index].bytesPerSecond
                items[index].bytesPerSecond = previous == 0 ? instant : previous * 0.7 + instant * 0.3
                rateClock[id] = (now, received)
            }
        } else {
            rateClock[id] = (now, received)
        }

        // Model rows also push their fraction into the provider's state
        // stream so the Setup overlay / model status show real progress.
        if items[index].kind == .model, let onProgress = modelProgressCallbacks[id] {
            onProgress(items[index].fractionComplete)
        }
    }

    /// The OS reported that a task is parked waiting for connectivity. Flip
    /// the row to an explicit state so the UI can say "Waiting for network"
    /// rather than a frozen 0%. Bytes resume (and the row returns to
    /// `.downloading`) as soon as a `didWriteData` callback lands.
    func waitingForNetwork(taskID: Int, label: TaskLabel?) {
        guard let id = itemID(taskID: taskID, label: label),
              let index = items.firstIndex(where: { $0.id == id }),
              items[index].state != .paused else { return }
        items[index].state = .waitingForNetwork
        items[index].bytesPerSecond = 0
        if items[index].kind == .model, let onWaiting = modelWaitingCallbacks[id] {
            onWaiting(true)
        }
        // Don't keep the device awake while it's waiting for a network —
        // there's no data flowing, so a plugged-in phone would just burn.
        updateSleepBlocker()
    }

    func finished(taskID: Int, label: TaskLabel?, staged: URL, destination: URL) {
        defer {
            try? FileManager.default.removeItem(at: staged)
            updateSleepBlocker()
        }
        guard let id = itemID(taskID: taskID, label: label),
              let row = items.first(where: { $0.id == id }),
              row.state == .downloading || row.state == .waitingForNetwork else { return }
        do {
            // Atomic rename on the destination volume. For model rows the
            // delegate already verified size + SHA-256 before staging, so a
            // truncated/corrupt GGUF is never moved over a working model.
            try ArchiveFilePolicy.commit(staged: staged, to: destination)
        }
        catch {
            failed(taskID: taskID, label: label, message: error.localizedDescription, resumeData: nil)
            return
        }
        retiredTaskIDs.insert(taskID)
        taskIDToItemID[taskID] = nil
        Self.deleteResumeData(id: id)
        if let index = items.firstIndex(where: { $0.id == id }) {
            items[index].state = .finished
            items[index].destination = destination
            items[index].bytesPerSecond = 0
            items[index].receivedBytes = max(items[index].receivedBytes, items[index].expectedBytes)
        }
        rateClock[id] = nil
        if row.kind == .model {
            modelProgressCallbacks[id] = nil
            modelWaitingCallbacks[id] = nil
            if let cont = modelCompletions.removeValue(forKey: id) {
                cont.resume(returning: destination)
            }
            NotificationCenter.default.post(name: Self.modelReadyNotification,
                                            object: nil,
                                            userInfo: ["url": destination, "id": id])
        } else {
            NotificationCenter.default.post(name: Self.fileReadyNotification,
                                            object: nil,
                                            userInfo: ["url": destination])
        }
    }

    func failed(taskID: Int, label: TaskLabel?, message: String, resumeData: Data?) {
        defer { updateSleepBlocker() }
        guard let id = itemID(taskID: taskID, label: label) else { return }
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        // A user-initiated pause also lands here (cancel error) — keep the
        // paused state it already has.
        if items[index].state == .paused { return }
        let isModel = items[index].kind == .model
        if let resumeData, let label {
            Self.writeResumeData(resumeData, id: id)
            Self.writeResumeLabel(label, id: id)
        }
        items[index].state = resumeData != nil ? .paused : .failed(message)
        retiredTaskIDs.insert(taskID)
        items[index].bytesPerSecond = 0
        rateClock[id] = nil
        // The row no longer owns a task; drop the mapping so a paused/failed
        // item can't be counted as active by `hasActiveDownloads`.
        taskIDToItemID[taskID] = nil
        // A model that can't resume (no partial bytes saved) is a hard
        // failure — tell the awaiting provider so `load()` surfaces the
        // error and the UI offers Retry. A resumable pause keeps waiting;
        // the provider's load stays pending until the user resumes.
        if isModel, resumeData == nil {
            modelProgressCallbacks[id] = nil
            modelWaitingCallbacks[id] = nil
            if let cont = modelCompletions.removeValue(forKey: id) {
                cont.resume(throwing: ZimDownloadError.failed(message))
            }
        }
    }

    fileprivate func backgroundEventsDrained() {
        backgroundEventsCompletionHandler?()
        backgroundEventsCompletionHandler = nil
    }

    private func itemID(taskID: Int, label: TaskLabel?) -> String? {
        guard !retiredTaskIDs.contains(taskID) else { return nil }
        if let id = taskIDToItemID[taskID] { return id }
        guard let label else { return nil }
        // An existing row owns a newer task or an intentional paused/failed
        // state. Only unknown rows can be restored from a task label.
        guard !items.contains(where: { $0.id == label.id }) else { return nil }
        // Task from a previous process (background relaunch): adopt it now.
        taskIDToItemID[taskID] = label.id
        if !items.contains(where: { $0.id == label.id }), let url = URL(string: label.urlString) {
            items.append(Self.makeItem(label: label, url: url))
        }
        return label.id
    }

    /// Build an item from a persisted task label, carrying the model-only
    /// fields (kind, SHA-256, cache destination) so a relaunch restores the
    /// transfer exactly where it left off.
    private nonisolated static func makeItem(label: TaskLabel, url: URL) -> Item {
        var item = Item(id: label.id, title: label.title, url: url,
                        expectedBytes: label.expectedBytes)
        if let kind = label.kind, let parsed = TransferKind(rawValue: kind) {
            item.kind = parsed
        }
        item.sha256 = label.sha256
        item.destination = label.destPath.map { URL(fileURLWithPath: $0) }
        return item
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
    /// `"zim"` or `"model"`. Absent on labels written before model support,
    /// so old resume blobs still decode (and default to `.zim`).
    let kind: String?
    /// Model rows only: expected SHA-256 of the finished weights.
    let sha256: String?
    /// Model rows only: the cache slot the finished file is moved into.
    let destPath: String?

    var encoded: String {
        var payload: [String: Any] = ["id": id, "title": title,
                                      "url": urlString, "bytes": expectedBytes]
        if let kind { payload["kind"] = kind }
        if let sha256 { payload["sha256"] = sha256 }
        if let destPath { payload["destPath"] = destPath }
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let text = String(data: data, encoding: .utf8) else { return id }
        return text
    }

    init(id: String, title: String, urlString: String, expectedBytes: Int64,
         kind: String? = nil, sha256: String? = nil, destPath: String? = nil) {
        self.id = id
        self.title = title
        self.urlString = urlString
        self.expectedBytes = expectedBytes
        self.kind = kind
        self.sha256 = sha256
        self.destPath = destPath
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
        self.kind = payload["kind"] as? String
        self.sha256 = payload["sha256"] as? String
        self.destPath = payload["destPath"] as? String
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
        let fm = FileManager.default
        do {
            if label?.kind == ZimDownloadManager.TransferKind.model.rawValue {
                // Model weights: stage under the cache slot's own directory
                // (same volume, so the final rename is atomic), and verify
                // size + SHA-256 here on the session queue before the file
                // can ever replace a working model.
                guard let destPath = label?.destPath, destPath.hasPrefix("/") else {
                    throw CocoaError(.fileNoSuchFile)
                }
                let destination = URL(fileURLWithPath: destPath)
                let statusCode = (downloadTask.response as? HTTPURLResponse)?.statusCode
                guard statusCode == 200 || statusCode == 206 else {
                    throw URLError(.badServerResponse)
                }
                let expected = label?.expectedBytes ?? 0
                if expected > 0 {
                    let got = (try? fm.attributesOfItem(atPath: location.path)[.size] as? Int64) ?? 0
                    guard got == expected else {
                        throw ZimDownloadError.sizeMismatch(expected: expected, got: got)
                    }
                }
                if let sha256 = label?.sha256 {
                    let digest = try ZimDownloadManager.sha256Hex(of: location)
                    guard digest.lowercased() == sha256.lowercased() else {
                        throw ZimDownloadError.checksumMismatch(expected: sha256, got: digest)
                    }
                }
                let dir = destination.deletingLastPathComponent()
                try fm.createDirectory(at: dir, withIntermediateDirectories: true)
                let staged = dir.appendingPathComponent(".download-\(UUID().uuidString).partial")
                try fm.moveItem(at: location, to: staged)
                Task { @MainActor [weak manager] in
                    guard let manager else { try? fm.removeItem(at: staged); return }
                    manager.finished(taskID: taskID, label: label, staged: staged, destination: destination)
                }
                return
            }

            // ZIM archive: validate the archive before it can replace an
            // existing library file, then stage and hand off.
            let filename = downloadTask.originalRequest?.url?.lastPathComponent
                ?? URL(string: label?.urlString ?? "")?.lastPathComponent
                ?? location.lastPathComponent
            guard let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask,
                                         appropriateFor: nil, create: true) else {
                throw CocoaError(.fileNoSuchFile)
            }
            try ArchiveFilePolicy.validateDownload(location,
                statusCode: (downloadTask.response as? HTTPURLResponse)?.statusCode,
                filename: filename)
            // Opening validates archive tables before an existing file can
            // be replaced. Transport/header checks alone are not enough.
            _ = try LibzimReader(url: location)
            let destination = docs.appendingPathComponent(filename)
            let staged = docs.appendingPathComponent(".download-\(UUID().uuidString).partial")
            try fm.moveItem(at: location, to: staged)
            Task { @MainActor [weak manager] in
                guard let manager else { try? fm.removeItem(at: staged); return }
                manager.finished(taskID: taskID, label: label, staged: staged, destination: destination)
            }
        } catch {
            let filename = downloadTask.originalRequest?.url?.lastPathComponent
                ?? URL(string: label?.urlString ?? "")?.lastPathComponent
                ?? location.lastPathComponent
            let message = "Couldn't save \(filename): \(error.localizedDescription)"
            Task { @MainActor [weak manager] in
                manager?.failed(taskID: taskID, label: label, message: message, resumeData: nil)
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    isWaitingForConnectivity waiting: Bool) {
        // Surface the OS's "offline" pause as an explicit UI state instead of
        // a frozen 0% spinner, so the user sees "Waiting for network" and
        // knows the transfer will resume on its own.
        guard waiting else { return }
        let label = TaskLabel(taskDescription: task.taskDescription)
        let taskID = task.taskIdentifier
        Task { @MainActor [weak manager] in
            manager?.waitingForNetwork(taskID: taskID, label: label)
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
        let message = nsError.localizedDescription
        Task { @MainActor [weak manager] in
            manager?.failed(taskID: taskID, label: label, message: message,
                            resumeData: resumeData)
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

// MARK: - Model transfer descriptor

/// What the background downloader needs to fetch a model's weights and land
/// them in the provider's cache slot. Built by a `LlamaCppProvider` when its
/// GGUF isn't already cached.
struct ModelDownloadSpec {
    let id: String            // stable provider id (matches `ModelProvider.id`)
    let title: String         // display name, surfaced in the download row
    let url: URL              // HuggingFace resolve URL
    let expectedBytes: Int64  // publisher-pinned byte count (0 = unknown)
    let sha256: String?       // pinned hex SHA-256 (verified before install)
    let destination: URL      // `<caches>/huggingface/hub/.../<file>.gguf`
}

/// Errors surfaced by model transfers, with user-facing messages.
enum ZimDownloadError: Error, LocalizedError {
    case insufficientStorage(needed: Int64, available: Int64, title: String)
    case sizeMismatch(expected: Int64, got: Int64)
    case checksumMismatch(expected: String, got: String)
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .insufficientStorage(let needed, let available, let title):
            return "Not enough storage for \(title). It needs "
                + "\(ZimDownloadManager.bytes(needed)) but only "
                + "\(ZimDownloadManager.bytes(available)) is free. Free up space or pick a smaller model."
        case .sizeMismatch(let expected, let got):
            return "Model download size mismatch (\(got)/\(expected) bytes) — the file was truncated. Retrying…"
        case .checksumMismatch(let expected, let got):
            return "Model download checksum mismatch (got \(got.prefix(12))…, expected \(expected.prefix(12))…) — the file was corrupted. Retrying…"
        case .failed(let message):
            return message
        }
    }
}
