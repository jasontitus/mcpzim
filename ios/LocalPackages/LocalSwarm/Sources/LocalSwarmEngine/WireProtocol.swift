import Foundation

/// The on-the-wire message set. Control messages are small and JSON-friendly;
/// bitfields and chunk payloads use compact binary encodings so a 1 MiB chunk
/// is never base64-inflated.
///
/// Frame layout: `[UInt32 frameLength][UInt8 type][body...]` (big-endian length).
public enum Message {
    case handshake(peerID: String, swarmID: String)
    case manifestRequest(swarmID: String)
    case manifestResponse(SwarmManifest)
    case bitfield(swarmID: String, bits: [Bool])
    case have(swarmID: String, chunkIndex: Int)
    case chunkRequest(swarmID: String, chunkIndex: Int)
    case chunkResponse(swarmID: String, chunkIndex: Int, data: Data)
    case auth(swarmID: String, token: String)
}

extension Message {
    /// The swarm a message is scoped to, when it carries one. Used to reject
    /// cross-swarm protocol confusion from a buggy or hostile peer.
    var swarmID: String? {
        switch self {
        case let .handshake(_, sid): return sid
        case let .manifestRequest(sid): return sid
        case let .bitfield(sid, _): return sid
        case let .have(sid, _): return sid
        case let .chunkRequest(sid, _): return sid
        case let .chunkResponse(sid, _, _): return sid
        case let .auth(sid, _): return sid
        case .manifestResponse: return nil
        }
    }
}

enum MessageType: UInt8 {
    case handshake = 1
    case manifestRequest = 2
    case manifestResponse = 3
    case bitfield = 4
    case have = 5
    case chunkRequest = 6
    case chunkResponse = 7
    case auth = 8
}

enum WireError: Error {
    case malformed
    case unknownType(UInt8)
}

enum Wire {
    /// Encodes a complete frame including the 4-byte length prefix.
    static func encode(_ message: Message) throws -> Data {
        var body = ByteWriter()
        switch message {
        case let .handshake(peerID, swarmID):
            body.u8(MessageType.handshake.rawValue)
            body.string(peerID)
            body.string(swarmID)
        case let .manifestRequest(swarmID):
            body.u8(MessageType.manifestRequest.rawValue)
            body.string(swarmID)
        case let .manifestResponse(manifest):
            body.u8(MessageType.manifestResponse.rawValue)
            let json = try JSONEncoder().encode(manifest)
            body.u32(UInt32(json.count))
            body.raw(json)
        case let .bitfield(swarmID, bits):
            body.u8(MessageType.bitfield.rawValue)
            body.string(swarmID)
            let packed = packBits(bits)
            body.u32(UInt32(bits.count))
            body.u32(UInt32(packed.count))
            body.raw(packed)
        case let .have(swarmID, chunkIndex):
            body.u8(MessageType.have.rawValue)
            body.string(swarmID)
            body.u32(UInt32(chunkIndex))
        case let .chunkRequest(swarmID, chunkIndex):
            body.u8(MessageType.chunkRequest.rawValue)
            body.string(swarmID)
            body.u32(UInt32(chunkIndex))
        case let .chunkResponse(swarmID, chunkIndex, data):
            body.u8(MessageType.chunkResponse.rawValue)
            body.string(swarmID)
            body.u32(UInt32(chunkIndex))
            body.u32(UInt32(data.count))
            body.raw(data)
        case let .auth(swarmID, token):
            body.u8(MessageType.auth.rawValue)
            body.string(swarmID)
            body.string(token)
        }

        var frame = ByteWriter()
        frame.u32(UInt32(body.data.count))
        frame.raw(body.data)
        return frame.data
    }

    /// Decodes a frame body (everything after the length prefix).
    static func decode(body: Data) throws -> Message {
        var reader = ByteReader(body)
        let rawType = try reader.u8()
        guard let type = MessageType(rawValue: rawType) else { throw WireError.unknownType(rawType) }
        switch type {
        case .handshake:
            return .handshake(peerID: try reader.string(), swarmID: try reader.string())
        case .manifestRequest:
            return .manifestRequest(swarmID: try reader.string())
        case .manifestResponse:
            let length = Int(try reader.u32())
            let json = try reader.take(length)
            let manifest = try JSONDecoder().decode(SwarmManifest.self, from: json)
            return .manifestResponse(manifest)
        case .bitfield:
            let swarmID = try reader.string()
            let bitCount = Int(try reader.u32())
            let packedLength = Int(try reader.u32())
            // Bound the bool allocation and require the packed length to match the
            // claimed bit count, so a peer can't force a huge/garbage allocation.
            guard bitCount >= 0, bitCount <= SwarmManifest.maxChunks,
                  packedLength == (bitCount + 7) / 8 else { throw WireError.malformed }
            let packed = try reader.take(packedLength)
            return .bitfield(swarmID: swarmID, bits: unpackBits(packed, count: bitCount))
        case .have:
            return .have(swarmID: try reader.string(), chunkIndex: Int(try reader.u32()))
        case .chunkRequest:
            return .chunkRequest(swarmID: try reader.string(), chunkIndex: Int(try reader.u32()))
        case .chunkResponse:
            let swarmID = try reader.string()
            let chunkIndex = Int(try reader.u32())
            let length = Int(try reader.u32())
            let data = try reader.take(length)
            return .chunkResponse(swarmID: swarmID, chunkIndex: chunkIndex, data: data)
        case .auth:
            return .auth(swarmID: try reader.string(), token: try reader.string())
        }
    }

    // MARK: - Bit packing

    static func packBits(_ bits: [Bool]) -> Data {
        var data = Data(count: (bits.count + 7) / 8)
        for (i, bit) in bits.enumerated() where bit {
            data[i / 8] |= UInt8(1 << (7 - (i % 8)))
        }
        return data
    }

    static func unpackBits(_ data: Data, count: Int) -> [Bool] {
        let bytes = [UInt8](data)
        var bits = [Bool](repeating: false, count: count)
        for i in 0..<count {
            let byteIndex = i / 8
            guard byteIndex < bytes.count else { break }
            bits[i] = (bytes[byteIndex] & UInt8(1 << (7 - (i % 8)))) != 0
        }
        return bits
    }
}

// MARK: - Byte helpers (alignment-safe, big-endian)

struct ByteWriter {
    private(set) var data = Data()

    mutating func u8(_ value: UInt8) { data.append(value) }

    mutating func u32(_ value: UInt32) {
        data.append(UInt8((value >> 24) & 0xff))
        data.append(UInt8((value >> 16) & 0xff))
        data.append(UInt8((value >> 8) & 0xff))
        data.append(UInt8(value & 0xff))
    }

    mutating func raw(_ bytes: Data) { data.append(bytes) }

    mutating func string(_ value: String) {
        let bytes = Data(value.utf8)
        u32(UInt32(bytes.count))
        raw(bytes)
    }
}

struct ByteReader {
    private let bytes: [UInt8]
    private var offset = 0

    init(_ data: Data) { bytes = [UInt8](data) }

    mutating func u8() throws -> UInt8 {
        guard offset < bytes.count else { throw WireError.malformed }
        defer { offset += 1 }
        return bytes[offset]
    }

    mutating func u32() throws -> UInt32 {
        guard offset + 4 <= bytes.count else { throw WireError.malformed }
        let value = (UInt32(bytes[offset]) << 24)
            | (UInt32(bytes[offset + 1]) << 16)
            | (UInt32(bytes[offset + 2]) << 8)
            | UInt32(bytes[offset + 3])
        offset += 4
        return value
    }

    mutating func take(_ count: Int) throws -> Data {
        guard count >= 0, offset + count <= bytes.count else { throw WireError.malformed }
        defer { offset += count }
        return Data(bytes[offset..<offset + count])
    }

    mutating func string() throws -> String {
        let count = Int(try u32())
        let data = try take(count)
        return String(decoding: data, as: UTF8.self)
    }
}
