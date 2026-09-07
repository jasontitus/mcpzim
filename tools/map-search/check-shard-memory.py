#!/usr/bin/env python3
"""Compare the old decoded cache with the shipping bounded parser.

Argument: directory containing be-0.json through be-f.json extracted from the
reported California ZIM. Runs each implementation in a fresh process. Does not
include app/LLM or libzim decompression memory; this isolates JSON/cache overhead.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
shards = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="zimfo-shard-memory-") as work:
    work = Path(work)
    main = work / "main.swift"
    main.write_text((root / "tools/map-search/ShardMemoryBench.swift").read_text())
    binary = work / "bench"
    subprocess.run(["swiftc", "-O", str(root / "swift/Sources/MCPZimKit/FilteredPlaceJSON.swift"),
                    str(main), "-o", str(binary)], check=True)
    outputs = []
    for mode in ["old", "new"]:
        output = subprocess.check_output([str(binary), mode, str(shards)], text=True).splitlines()
        outputs.append(output)
        print(output[0])
    assert outputs[0][1:] == outputs[1][1:], "Search result names differ"
    print("Identical matching names: yes")
