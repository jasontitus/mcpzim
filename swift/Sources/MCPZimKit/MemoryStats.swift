// SPDX-License-Identifier: MIT
//
// Lightweight resident-memory probe. Reads `task_vm_info.phys_footprint`
// (the same number Xcode's memory gauge shows) so debug logs can attribute
// spikes to specific ZIM reads or generation steps.
//
// `phys_footprint` is what iOS uses to decide which processes to jetsam, so
// it's the metric that actually matters for "am I about to get killed?"

import Foundation
import Darwin.Mach

public enum MemoryStats {
    /// Resident footprint in bytes. Returns 0 if the sysctl fails (rare).
    public static func physFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { typed in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), typed, &count)
            }
        }
        return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
    }

    /// Human-readable megabytes (base-10, matching most memory UIs).
    public static func physFootprintMB() -> Double {
        Double(physFootprintBytes()) / 1_048_576
    }

    /// Formatted `"123.4 MB"` string for direct use in logs.
    public static func formatted() -> String {
        String(format: "%.1f MB", physFootprintMB())
    }
}
