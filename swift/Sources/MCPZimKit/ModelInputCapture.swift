import Foundation
import CryptoKit

/// Explicitly enabled by the evaluation CLI. Records model inputs, never
/// calibration activations. A quantized provider may supply useful prompts,
/// but all reference activations must be collected separately using BF16.
public final class ModelInputCapture: @unchecked Sendable {
    public static let shared = ModelInputCapture()
    private let lock = NSLock()
    private var directory: URL?
    private var metadata: [String: String] = [:]
    private var conversation: String?
    private var turn: Int?
    private var ordinal = 0
    private var failure: String?

    public init() {}

    /// Require a new directory so a failed/repeated run cannot overwrite or
    /// silently mix with a previous dataset. No environment-based activation.
    public func start(directory: URL, metadata: [String: String]) throws {
        lock.lock(); defer { lock.unlock() }
        guard self.directory == nil else { throw CaptureError.alreadyStarted }
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            throw CaptureError.directoryExists
        }
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        let marker = directory.appendingPathComponent("capture-run.json")
        var runMetadata = metadata
        runMetadata["status"] = "running"
        let run = try JSONEncoder().encode(runMetadata)
        try run.write(to: marker, options: [.withoutOverwriting])
        self.directory = directory
        self.metadata = metadata
        conversation = nil; turn = nil; ordinal = 0; failure = nil
    }

    public func setCase(conversation: String, turn: Int) {
        lock.lock(); defer { lock.unlock() }
        self.conversation = conversation; self.turn = turn
    }

    public func stop() {
        lock.lock(); defer { lock.unlock() }
        directory = nil
    }

    /// A completed capture can contain failed answers: this status certifies
    /// input recording only, not answer quality or BF16 reference extraction.
    public func finish() throws {
        lock.lock(); defer { lock.unlock() }
        guard let directory else { return }
        if let failure { throw CaptureError.writeFailed(failure) }
        var result = metadata
        result["status"] = "completed"
        result["invocationCount"] = String(ordinal)
        try JSONEncoder().encode(result).write(
            to: directory.appendingPathComponent("capture-run.json"), options: .atomic)
        self.directory = nil
    }

    public var recordedCount: Int {
        lock.lock(); defer { lock.unlock() }
        return ordinal
    }

    public func checkHealth() throws {
        lock.lock(); defer { lock.unlock() }
        if let failure { throw CaptureError.writeFailed(failure) }
    }

    public struct Record: Codable, Sendable {
        public let schemaVersion: Int
        public let ordinal: Int
        public let conversationID: String
        public let turn: Int
        public let modelID: String
        public let runtime: String
        public let prompt: String
        public let promptSHA256: String
        public let tokenIDs: [Int32]
        public let sampler: [String: Double]
        public let stopSequences: [String]
        public let metadata: [String: String]
    }

    /// Call after tokenization/context validation, before prefill or cache
    /// suffix selection. Always stores the full prompt, including reused text.
    /// Failure is sticky because ChatSession can catch generation errors.
    public func record(modelID: String, runtime: String, prompt: String,
                       tokenIDs: [Int32], sampler: [String: Double],
                       stopSequences: [String]) throws {
        lock.lock(); defer { lock.unlock() }
        guard let directory else { return }
        do {
            if let failure { throw CaptureError.writeFailed(failure) }
            guard let conversation, let turn, turn >= 0,
                  !tokenIDs.isEmpty else { throw CaptureError.missingContext }
            let record = Record(
                schemaVersion: 1, ordinal: ordinal,
                conversationID: conversation, turn: turn,
                modelID: modelID, runtime: runtime, prompt: prompt,
                promptSHA256: SHA256.hash(data: Data(prompt.utf8))
                    .map { String(format: "%02x", $0) }.joined(),
                tokenIDs: tokenIDs, sampler: sampler,
                stopSequences: stopSequences, metadata: metadata)
            let data = try JSONEncoder().encode(record)
            try data.write(to: directory.appendingPathComponent(
                String(format: "invocation-%06d.json", ordinal)),
                options: [.atomic])
            ordinal += 1
        } catch {
            failure = String(describing: error)
            throw error
        }
    }

    public enum CaptureError: Error {
        case alreadyStarted, directoryExists, missingContext, writeFailed(String)
    }
}
