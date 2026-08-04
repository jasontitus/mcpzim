import Foundation
import Network

/// Tracks the live set of network interfaces so a data connection can be steered
/// onto the direct peer-to-peer link (AWDL) instead of the infrastructure router.
///
/// `includePeerToPeer` alone isn't enough: when both devices sit on the same
/// Wi-Fi, the system prefers the router path (`en0`, tens of Mbps) over the
/// direct AWDL link (`awdl0`, ~Gbps). Prohibiting the infrastructure interfaces
/// on the connection forces AWDL to come up.
final class InterfaceTracker {
    static let shared = InterfaceTracker()

    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "com.localswarm.ifaces")
    private let lock = NSLock()
    private var current: [NWInterface] = []
    private var logged = false

    private init() {
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self = self else { return }
            self.lock.lock()
            self.current = path.availableInterfaces
            let first = !self.logged
            self.logged = true
            self.lock.unlock()
            if first {
                let desc = path.availableInterfaces.map { "\($0.name)(\($0.type))" }.joined(separator: ", ")
                swarmDiag("interfaces available: [\(desc)] (status \(path.status))")
            }
        }
        monitor.start(queue: queue)
    }

    func start() { /* touching .shared starts the monitor */ }

    /// Interfaces to prohibit when forcing a direct link: router Wi-Fi (`en*`),
    /// wired Ethernet, and cellular. The direct links (`awdl*`, `llw*`) are never
    /// included, so prohibiting these leaves AWDL as the only path.
    func infrastructureInterfaces() -> [NWInterface] {
        lock.lock(); defer { lock.unlock() }
        return current.filter { iface in
            let name = iface.name
            if name.hasPrefix("awdl") || name.hasPrefix("llw") { return false }
            return name.hasPrefix("en") || iface.type == .wiredEthernet || iface.type == .cellular
        }
    }
}
