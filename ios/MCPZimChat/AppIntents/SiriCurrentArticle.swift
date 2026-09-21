// SPDX-License-Identifier: MIT

import AppIntents
import CoreTransferable
import Foundation
import MCPZimKit
import Observation
import SwiftUI
import UniformTypeIdentifiers
#if os(macOS)
import AppKit
#endif

/// Exact source captured when the discussion or preview resolves an article.
/// No title search, transcript, or additional discussion sources are exported.
struct SiriArticleReference: Equatable, Sendable {
    let article: OfflineArticle
    let libraryVersion: String

    static func resolved(_ result: [String: Any], version: String) -> Self? {
        guard result["error"] == nil,
              let zim = result["zim"] as? String, !zim.isEmpty,
              let path = result["path"] as? String, !path.isEmpty, path.count <= 2048,
              let title = result["title"] as? String, !title.isEmpty, title.count <= 500
        else { return nil }
        return .init(article: .init(zim: zim, path: path, title: title), libraryVersion: version)
    }
}

/// This entity is deliberately separate from saved Shortcuts article outputs.
/// Its opaque ID resolves only while this specific onscreen context is live.
struct ZimfoCurrentArticleEntity: AppEntity, Transferable {
    static var typeDisplayRepresentation = TypeDisplayRepresentation(name: "Current offline article")
    static var defaultQuery = ZimfoCurrentArticleQuery()
    let id: String
    @Property(title: "Title") var title: String
    @Property(title: "Archive") var archive: String
    var displayRepresentation: DisplayRepresentation {
        .init(title: "\(title)", subtitle: "\(archive)")
    }
    init(id: String, title: String, archive: String) {
        self.id = id; self.title = title; self.archive = archive
    }
    static var transferRepresentation: some TransferRepresentation {
        DataRepresentation(exportedContentType: .plainText) { entity in
            try await SiriCurrentArticleContext.shared.export(id: entity.id)
        }
    }
}

struct ZimfoCurrentArticleQuery: EntityQuery {
    @MainActor
    func entities(for identifiers: [String]) async throws -> [ZimfoCurrentArticleEntity] {
        let trace = SiriInvocation("current_article_query")
        defer { trace.end() }
        try Task.checkCancellation()
        return Array(Set(identifiers.prefix(5))).compactMap { SiriCurrentArticleContext.shared.entity(id: $0) }
    }
    func suggestedEntities() async throws -> [ZimfoCurrentArticleEntity] { [] }
}

/// One in-memory lease, zero indexed items. Replacing/closing the view revokes
/// old IDs and in-flight exports. System copies already delivered cannot be
/// recalled; clearing controls future access, not Siri's conversation history.
@MainActor @Observable
final class SiriCurrentArticleContext {
    static let shared = SiriCurrentArticleContext()
    static let activityType = "com.tiltastech.zimfo.current-article"
    private(set) var current: ZimfoCurrentArticleEntity?
    private var reference: SiriArticleReference?
    private var owner: UUID?
    @ObservationIgnored private var activity: NSUserActivity?
    @ObservationIgnored private var eligible: () -> Bool = { false }
    #if os(macOS)
    @ObservationIgnored private weak var publishedWindow: NSWindow?
    @ObservationIgnored var keyWindow: @MainActor () -> NSWindow? = { NSApp?.keyWindow }
    @ObservationIgnored private var focusObserver: NSObjectProtocol?
    #endif
    @ObservationIgnored private let fingerprint: @MainActor () -> String
    @ObservationIgnored private let loader: (SiriArticleReference) async throws -> Data

    init(fingerprint: @escaping @MainActor () -> String = { ZimfoRunner.libraryFingerprint() },
         loader: @escaping (SiriArticleReference) async throws -> Data = { ref in
             let runner = try await ZimfoRunner.load()
             guard runner.libraryVersion == ref.libraryVersion else { throw OfflineKnowledge.Failure.staleArticle }
             // document validates exact path/title and size before text export.
             return try await OfflineKnowledge(service: runner.service).document(article: ref.article)
         }) {
        self.fingerprint = fingerprint
        self.loader = loader
        #if os(macOS)
        focusObserver = NotificationCenter.default.addObserver(forName: NSWindow.didBecomeKeyNotification, object: nil, queue: .main) { [weak self] note in
            MainActor.assumeIsolated {
                guard let self, let window = note.object as? NSWindow,
                      self.current != nil, self.publishedWindow !== window else { return }
                // Also revoke for settings/empty windows with no article view.
                // Comparing windows avoids chat/preview callback-order races.
                self.revoke(reason: "different_window")
            }
        }
        #endif
    }

    deinit {
        #if os(macOS)
        if let focusObserver { NotificationCenter.default.removeObserver(focusObserver) }
        #endif
    }

    func publish(_ source: SiriArticleReference, owner newOwner: UUID, eligible: @escaping () -> Bool) {
        guard eligible(), source.libraryVersion == fingerprint() else { clear(owner: newOwner); return }
        if owner == newOwner, reference == source, current != nil { self.eligible = eligible; return }
        revoke(reason: "source_replaced")
        owner = newOwner
        #if os(macOS)
        publishedWindow = keyWindow()
        #endif
        reference = source
        self.eligible = eligible
        current = .init(id: UUID().uuidString, title: source.article.title, archive: source.article.zim)
        let trace = SiriInvocation("current_article_context")
        trace.stage("published")
        trace.end()
    }

    func clear(owner expectedOwner: UUID, reason: String = "owner_cleared") {
        guard owner == expectedOwner else { return }
        revoke(reason: reason)
    }

    private func revoke(reason: String) {
        if current != nil {
            let trace = SiriInvocation("current_article_context")
            trace.stage("revoked_" + reason)
            trace.end()
        }
        if #available(iOS 18.2, macOS 15.2, *) { activity?.appEntityIdentifier = nil }
        activity?.resignCurrent()
        activity?.invalidate()
        activity = nil
        current = nil
        reference = nil
        owner = nil
        #if os(macOS)
        publishedWindow = nil
        #endif
        eligible = { false }
    }

    func entity(id: String) -> ZimfoCurrentArticleEntity? {
        guard id.count <= 64, let current, id == current.id,
              let reference, eligible(), reference.libraryVersion == fingerprint() else { return nil }
        return current
    }

    func entity(owner: UUID) -> ZimfoCurrentArticleEntity? {
        guard self.owner == owner, let current else { return nil }
        return entity(id: current.id)
    }

    func configure(_ activity: NSUserActivity, id: String) {
        let trace = SiriInvocation("current_article_activity")
        defer { trace.end() }
        guard #available(iOS 18.2, macOS 15.2, *), let entity = entity(id: id) else {
            activity.invalidate()
            trace.stage("configuration_rejected")
            return
        }
        self.activity = activity
        activity.title = entity.title
        activity.isEligibleForSearch = false
        activity.isEligibleForPublicIndexing = false
        #if os(iOS)
        activity.isEligibleForPrediction = false
        #endif
        activity.isEligibleForHandoff = false
        activity.appEntityIdentifier = EntityIdentifier(for: entity)
        trace.stage("entity_associated")
        // No text in userInfo, no web fallback, no contentAttributeSet/index.
    }

    func export(id: String) async throws -> Data {
        let trace = SiriInvocation("current_article_export")
        defer { trace.end() }
        do {
            try Task.checkCancellation()
            guard entity(id: id) != nil, let reference else { throw OfflineKnowledge.Failure.staleArticle }
            let data = try await loader(reference)
            try Task.checkCancellation()
            guard entity(id: id) != nil, self.reference == reference else { throw OfflineKnowledge.Failure.staleArticle }
            guard !data.isEmpty, data.count <= OfflineKnowledge.maxDocumentBytes else { throw OfflineKnowledge.Failure.tooLarge }
            trace.stage("document_bytes", count: data.count)
            return data
        } catch { trace.failed(error); throw error }
    }
}

/// A view owns its lease. Hiding one window cannot revoke another window's
/// article. A Siri overlay can make the scene inactive, so only background
/// and actual view visibility revoke it; no app-wide last-topic fallback.
private struct SiriCurrentArticleModifier: ViewModifier {
    let source: SiriArticleReference?
    let visible: Bool
    let eligible: () -> Bool
    @Environment(\.scenePhase) private var scenePhase
    @State private var owner = UUID()
    @State private var appeared = false
    #if os(macOS)
    @State private var windowCanPublish = false
    #endif

    func body(content: Content) -> some View {
        content
            #if os(macOS)
            .background(CurrentArticleWindowObserver(onKey: {
                windowCanPublish = true
                refresh()
            }, onResignKey: {
                // Siri can take key focus; retain an existing visible lease,
                // but don't let changes in a non-key window steal ownership.
                windowCanPublish = false
            }, onHide: {
                windowCanPublish = false
                SiriCurrentArticleContext.shared.clear(owner: owner, reason: "window_hidden")
            }))
            #endif
            .userActivity(SiriCurrentArticleContext.activityType,
                          element: SiriCurrentArticleContext.shared.entity(owner: owner)) { entity, activity in
                SiriCurrentArticleContext.shared.configure(activity, id: entity.id)
            }
            .onAppear { appeared = true; refresh() }
            .onDisappear { appeared = false; SiriCurrentArticleContext.shared.clear(owner: owner, reason: "view_disappeared") }
            .onChange(of: source) { _, _ in refresh() }
            .onChange(of: visible) { _, _ in refresh() }
            .onChange(of: scenePhase) { _, _ in refresh() }
    }

    private func refresh() {
        if source != nil || SiriCurrentArticleContext.shared.current != nil {
            let trace = SiriInvocation("current_article_visibility")
            switch scenePhase {
            case .active: trace.stage("scene_active")
            case .inactive: trace.stage("scene_inactive")
            case .background: trace.stage("scene_background")
            @unknown default: trace.stage("scene_unknown")
            }
            trace.stage("visible", count: visible ? 1 : 0)
            trace.stage("appeared", count: appeared ? 1 : 0)
            trace.stage("has_source", count: source == nil ? 0 : 1)
            trace.end()
        }
        guard appeared, visible, scenePhase != .background, let source else {
            let reason = !appeared ? "view_not_appeared"
                : !visible ? "view_hidden"
                : scenePhase == .background ? "scene_background" : "no_source"
            SiriCurrentArticleContext.shared.clear(owner: owner, reason: reason)
            return
        }
        #if os(macOS)
        guard windowCanPublish else { return }
        #endif
        SiriCurrentArticleContext.shared.publish(source, owner: owner, eligible: eligible)
    }
}

extension View {
    func siriCurrentArticle(_ source: SiriArticleReference?, visible: Bool = true,
                            eligible: @escaping () -> Bool = { true }) -> some View {
        modifier(SiriCurrentArticleModifier(source: source, visible: visible, eligible: eligible))
    }
}

#if os(macOS)
/// Observe actual window focus: SwiftUI scenePhase alone doesn't distinguish
/// two Zimfo windows. Re-keying the remaining window restores its own source.
struct CurrentArticleWindowObserver: NSViewRepresentable {
    let onKey: () -> Void
    let onResignKey: () -> Void
    let onHide: () -> Void
    func makeNSView(context: Context) -> ObserverView { ObserverView() }
    func updateNSView(_ view: ObserverView, context: Context) {
        view.onKey = onKey; view.onResignKey = onResignKey; view.onHide = onHide
        view.attachIfNeeded()
    }
    static func dismantleNSView(_ view: ObserverView, coordinator: ()) { view.stop() }

    final class ObserverView: NSView {
        var onKey: () -> Void = {}
        var onResignKey: () -> Void = {}
        var onHide: () -> Void = {}
        private weak var observedWindow: NSWindow?
        private var observers: [NSObjectProtocol] = []
        override func viewDidMoveToWindow() { super.viewDidMoveToWindow(); attachIfNeeded() }
        func attachIfNeeded() {
            guard observedWindow !== window else { return }
            stop()
            guard let window else { return }
            observedWindow = window
            let center = NotificationCenter.default
            observers.append(center.addObserver(forName: NSWindow.didBecomeKeyNotification, object: window, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.onKey() }
            })
            observers.append(center.addObserver(forName: NSWindow.didResignKeyNotification, object: window, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.onResignKey() }
            })
            for name in [NSWindow.didMiniaturizeNotification, NSWindow.willCloseNotification] {
                observers.append(center.addObserver(forName: name, object: window, queue: .main) { [weak self] _ in
                    MainActor.assumeIsolated { self?.onHide() }
                })
            }
            observers.append(center.addObserver(forName: NSApplication.didHideNotification, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.onHide() }
            })
            // Do not mutate SwiftUI state during representable reconciliation.
            DispatchQueue.main.async { [weak self, weak window] in
                guard self?.window === window, window?.isKeyWindow == true else { return }
                self?.onKey()
            }
        }
        func stop() {
            for observer in observers { NotificationCenter.default.removeObserver(observer) }
            observers.removeAll(); observedWindow = nil
        }
        deinit { for observer in observers { NotificationCenter.default.removeObserver(observer) } }
    }
}
#endif
