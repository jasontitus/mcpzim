#!/usr/bin/env python3
"""Run the shipping checksum routine in isolation; fail if read buffers accumulate.

macOS only. Uses a sparse 512 MiB file, or a supplied model path. Compiles the
actual function from ZimDownloadManager.swift rather than a duplicate. Peak RSS
includes the whole helper process. A caller-level autorelease pool deliberately
reproduces the background-worker lifetime that triggered iOS launch jetsam.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
source = (root / "ios/MCPZimChat/Sharing/ZimDownloadManager.swift").read_text()
start = source.index("    nonisolated static func sha256Hex(")
end = source.index("    nonisolated static func bytes(", start)
with tempfile.TemporaryDirectory(prefix="zimfo-checksum-") as work:
    work = Path(work)
    if len(sys.argv) > 1:
        model = Path(sys.argv[1]).resolve()
    else:
        model = work / "fixture.bin"
        with model.open("wb") as output:
            output.truncate(512 * 1024 * 1024)
    program = 'import Foundation\nimport CryptoKit\nstruct Subject {\n'
    program += source[start:end] + '}\n'
    program += '''
let url = URL(fileURLWithPath: CommandLine.arguments[1])
try autoreleasepool {
    let digest = try Subject.sha256Hex(of: url)
    var usage = rusage()
    precondition(getrusage(RUSAGE_SELF, &usage) == 0)
    print("peak RSS: \\(usage.ru_maxrss) bytes; SHA-256: \\(digest)")
    precondition(usage.ru_maxrss < 64 * 1024 * 1024, "Checksum retained read buffers")
}
'''
    path = work / "main.swift"
    path.write_text(program)
    exe = work / "check"
    subprocess.run(["swiftc", "-module-cache-path", str(work / "modules"), str(path), "-o", str(exe)], check=True)
    subprocess.run([str(exe), str(model)], check=True)
