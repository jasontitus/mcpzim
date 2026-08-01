// SPDX-License-Identifier: MIT
//
// A* driving-time router for streetzim graphs. Cost and heuristic exactly
// mirror streetzim's JS viewer (resources/viewer/index.html `findRoute`) so
// results match the on-device map UI byte-for-byte.

import Foundation

public struct RoadSegment: Equatable, Sendable {
    public let name: String
    public let distanceMeters: Double
    public let durationSeconds: Double
}

public struct Route: Sendable {
    public let origin: (lat: Double, lon: Double)
    public let destination: (lat: Double, lon: Double)
    public let originNode: Int
    public let destinationNode: Int
    public let distanceMeters: Double
    public let durationSeconds: Double
    public let roads: [RoadSegment]
    public let polyline: [(lat: Double, lon: Double)]

    public var distanceKilometers: Double { distanceMeters / 1000 }
    public var durationMinutes: Double { durationSeconds / 60 }

    public var turnByTurn: [String] {
        roads.map { r in
            let name = r.name.isEmpty ? "(unnamed road)" : r.name
            return String(format: "%@ for %.2f km (~%.1f min)",
                          name,
                          r.distanceMeters / 1000,
                          r.durationSeconds / 60)
        }
    }
}

/// Driving-time A* search.
///
/// - Cost of a single edge (seconds) = `distance_m / (speed_kmh / 3.6)`.
/// - Admissible heuristic assumes no road is faster than the graph's true
///   max edge speed (capped at the JS viewer's 100 km/h), giving
///   `haversine / (ceiling/3.6)`.
public func aStar(graph: SZRGGraph, origin: Int, goal: Int) -> Route? {
    if origin == goal {
        return Route(
            origin: (graph.lat[origin], graph.lon[origin]),
            destination: (graph.lat[goal], graph.lon[goal]),
            originNode: origin,
            destinationNode: goal,
            distanceMeters: 0,
            durationSeconds: 0,
            roads: [],
            polyline: [(graph.lat[origin], graph.lon[origin])]
        )
    }

    let goalLat = graph.lat[goal]
    let goalLon = graph.lon[goal]
    // Tighter ceiling on slow (city) graphs = stronger h = fewer pops;
    // still admissible because no edge is faster than the graph max.
    let speedCeilingMps = min(100.0, max(1.0, graph.maxSpeedKmh)) / 3.6

    func heuristic(_ node: Int) -> Double {
        haversineMeters(graph.lat[node], graph.lon[node], goalLat, goalLon) / speedCeilingMps
    }

    // Node-indexed arrays instead of `[Int: …]` dictionaries. For a
    // 200K-node visit (cross-metro route) this drops ~80 MB of hash-
    // table overhead and keeps cache locality while still letting us
    // use `Double.infinity` / `-1` as the "unvisited" sentinel.
    // .infinity pre-fills a contiguous Float64 array — cheap on modern
    // ARM and lets the inner loop avoid dictionary hashing entirely.
    let n = graph.numNodes
    var gScore = [Double](repeating: .infinity, count: n)
    var cameFromPrev = [Int32](repeating: -1, count: n)
    var cameFromEdge = [Int32](repeating: -1, count: n)
    gScore[origin] = 0
    var open = MinHeap<QueueItem>()
    var counter = 0
    open.push(QueueItem(f: heuristic(origin), g: 0, tiebreaker: counter, node: origin))
    counter += 1

    while let current = open.pop() {
        if current.node == goal {
            return reconstructRoute(
                graph: graph, origin: origin, goal: goal,
                cameFromPrev: cameFromPrev, cameFromEdge: cameFromEdge
            )
        }
        let curG = gScore[current.node]
        // Guard against stale entries left in the heap after a better path was found.
        if current.g > curG { continue }

        let start = Int(graph.adjOffsets[current.node])
        let end = Int(graph.adjOffsets[current.node + 1])
        for e in start..<end {
            let target = Int(graph.edgeTargets[e])
            let dist = graph.edgeDistMeters[e]
            let speed = max(1.0, Double(graph.edgeSpeedKmh[e]))
            let edgeCost = dist * 3.6 / speed
            let tentative = curG + edgeCost
            if tentative < gScore[target] {
                gScore[target] = tentative
                cameFromPrev[target] = Int32(current.node)
                cameFromEdge[target] = Int32(e)
                open.push(QueueItem(f: tentative + heuristic(target), g: tentative, tiebreaker: counter, node: target))
                counter += 1
            }
        }
    }
    return nil
}

private func reconstructRoute(
    graph: SZRGGraph,
    origin: Int,
    goal: Int,
    cameFromPrev: [Int32],
    cameFromEdge: [Int32]
) -> Route {
    var reversed: [(prev: Int, edge: Int, this: Int)] = []
    var node = goal
    while node != origin {
        let prev = Int(cameFromPrev[node])
        let edge = Int(cameFromEdge[node])
        guard prev >= 0, edge >= 0 else { break }
        reversed.append((prev: prev, edge: edge, this: node))
        node = prev
    }
    reversed.reverse()

    var polyline: [(lat: Double, lon: Double)] = [(graph.lat[origin], graph.lon[origin])]
    var roads: [RoadSegment] = []
    var totalMeters = 0.0
    var totalSeconds = 0.0

    for step in reversed {
        let dist = graph.edgeDistMeters[step.edge]
        let speed = max(1.0, Double(graph.edgeSpeedKmh[step.edge]))
        let seconds = dist * 3.6 / speed
        totalMeters += dist
        totalSeconds += seconds

        let geomIdx = graph.edgeGeomIdx[step.edge]
        if geomIdx >= 0 && Int(geomIdx) < graph.geoms.count,
           !graph.geoms[Int(geomIdx)].isEmpty {
            let pts = graph.geoms[Int(geomIdx)]
            let prevLL = (graph.lat[step.prev], graph.lon[step.prev])
            let forward = distSq(pts.first!, prevLL) <= distSq(pts.last!, prevLL)
            let oriented = forward ? pts : Array(pts.reversed())
            polyline.append(contentsOf: oriented.dropFirst())
        } else {
            polyline.append((graph.lat[step.this], graph.lon[step.this]))
        }

        let name = graph.name(graph.edgeNameIdx[step.edge])
        if var last = roads.last, last.name == name {
            roads.removeLast()
            last = RoadSegment(
                name: last.name,
                distanceMeters: last.distanceMeters + dist,
                durationSeconds: last.durationSeconds + seconds
            )
            roads.append(last)
        } else {
            roads.append(RoadSegment(name: name, distanceMeters: dist, durationSeconds: seconds))
        }
    }

    return Route(
        origin: (graph.lat[origin], graph.lon[origin]),
        destination: (graph.lat[goal], graph.lon[goal]),
        originNode: origin,
        destinationNode: goal,
        distanceMeters: totalMeters,
        durationSeconds: totalSeconds,
        roads: roads,
        polyline: polyline
    )
}

// MARK: - Spatial (SZCI/SZRC) async A*
//
// Mirrors the streetzim JS viewer's `findRouteSpatialAStar` /
// `findRouteSpatialFiltered` (resources/viewer/index.html). Required for
// large-country ZIMs built with `--spatial-chunk-scale`: those ship no
// monolithic graph.bin (it wouldn't fit in mobile RAM) — the routing graph
// is a lazily-fetched cell grid behind `SpatialGraph`. Same 80 km/h
// heuristic ceiling, greedy weights, pop limits, and crow-distance
// thresholds as the viewer, so on-device routes match the map UI.

/// Nearest graph node to (lat, lon) by squared e7 distance — linear scan
/// over the eager node table (matches the JS `nearestNode`).
public func nearestNodeSpatial(index: SZCIIndex, lat: Double, lon: Double) -> Int {
    let latE7 = Int32((lat * 1e7).rounded())
    let lonE7 = Int32((lon * 1e7).rounded())
    let nodes = index.nodesScaled
    var best = -1
    var bestDist = Double.infinity
    var i = 0
    let n = nodes.count
    while i + 1 < n {
        // Int64: an Int32 diff traps on antimeridian-spanning data (>214.7°).
        let dlat = Double(Int64(nodes[i]) - Int64(latE7))
        let dlon = Double(Int64(nodes[i + 1]) - Int64(lonE7))
        let d = dlat * dlat + dlon * dlon
        if d < bestDist { bestDist = d; best = i / 2 }
        i += 2
    }
    return best
}

private struct SpatialPrevEdge {
    let source: Int
    let speedDist: UInt32
    let geomLocal: UInt32
    let nameIdx: UInt32
}

/// Core spatial A* (mirrors `findRouteSpatialAStar`, highwayOnly=false).
/// `greedyWeight` 1.0 = optimal; >1 inflates the heuristic for a faster,
/// slightly-suboptimal search on long routes. Returns nil if it exceeds
/// `popLimit` (caller retries greedier) or no path exists. Sparse Maps so
/// memory scales with nodes VISITED, not the graph's millions.
func aStarSpatial(
    graph: SpatialGraph, index: SZCIIndex,
    origin: Int, goal: Int,
    greedyWeight: Double, popLimit: Int
) async -> Route? {
    @inline(__always) func coord(_ node: Int) -> (lat: Double, lon: Double) {
        (Double(index.nodesScaled[node * 2]) / 1e7,
         Double(index.nodesScaled[node * 2 + 1]) / 1e7)
    }
    let goalC = coord(goal)
    let speedCeil = 80.0 / 3.6   // mirror the JS viewer's heuristic ceiling
    @inline(__always) func heur(_ node: Int) -> Double {
        let c = coord(node)
        return haversineMeters(c.lat, c.lon, goalC.lat, goalC.lon) / speedCeil
    }
    if origin == goal {
        let c = coord(origin)
        return Route(origin: c, destination: c, originNode: origin,
                     destinationNode: goal, distanceMeters: 0, durationSeconds: 0,
                     roads: [], polyline: [c])
    }

    var gScore: [Int: Double] = [origin: 0]
    var prevEdge: [Int: SpatialPrevEdge] = [:]
    var closed = Set<Int>()
    var open = MinHeap<QueueItem>()
    var counter = 0
    open.push(QueueItem(f: heur(origin), g: 0, tiebreaker: counter, node: origin)); counter += 1
    var pops = 0

    while let item = open.pop() {
        let current = item.node
        pops += 1
        if pops > popLimit { return nil }
        if current == goal { break }
        if closed.contains(current) { continue }
        let curG = gScore[current] ?? .infinity
        // Stale entry — a better path to this node was relaxed after the push.
        if item.g > curG { continue }
        closed.insert(current)
        let edges: [SpatialEdge]
        do { edges = try await graph.edgesOfNode(current) } catch { return nil }
        for e in edges {
            let target = Int(e.target)
            if closed.contains(target) { continue }
            let speed = Double(e.speedKmh)
            if speed == 0 { continue }
            let cost = e.distanceMeters * 3.6 / speed
            let tentative = curG + cost
            if tentative < (gScore[target] ?? .infinity) {
                gScore[target] = tentative
                prevEdge[target] = SpatialPrevEdge(
                    source: current, speedDist: e.speedDist,
                    geomLocal: e.geomLocal, nameIdx: e.nameIdx)
                let c = coord(target)
                let h = haversineMeters(c.lat, c.lon, goalC.lat, goalC.lon)
                    / speedCeil * greedyWeight
                open.push(QueueItem(f: tentative + h, g: tentative, tiebreaker: counter, node: target))
                counter += 1
            }
        }
    }

    guard let totalSeconds = gScore[goal], prevEdge[goal] != nil else { return nil }
    // Reconstruct goal → origin, then forward.
    var rev: [(pe: SpatialPrevEdge, node: Int)] = []
    var node = goal
    while node != origin {
        guard let pe = prevEdge[node] else { break }
        rev.append((pe, node))
        node = pe.source
    }
    rev.reverse()

    var polyline: [(lat: Double, lon: Double)] = [coord(origin)]
    var roads: [RoadSegment] = []
    var totalMeters = 0.0
    for (pe, thisNode) in rev {
        let speed = max(1.0, Double((pe.speedDist >> 24) & 0xFF))
        let dist = Double(pe.speedDist & 0x00FFFFFF) / 10.0
        let seconds = dist * 3.6 / speed
        totalMeters += dist
        if let pts = try? await graph.decodeGeomForEdge(
                sourceNode: pe.source, geomLocal: pe.geomLocal) {
            polyline.append(contentsOf: pts.dropFirst())
        } else {
            polyline.append(coord(thisNode))
        }
        let name = index.name(pe.nameIdx)
        if var last = roads.last, last.name == name {
            roads.removeLast()
            last = RoadSegment(name: last.name,
                               distanceMeters: last.distanceMeters + dist,
                               durationSeconds: last.durationSeconds + seconds)
            roads.append(last)
        } else {
            roads.append(RoadSegment(name: name, distanceMeters: dist,
                                     durationSeconds: seconds))
        }
    }

    return Route(origin: coord(origin), destination: coord(goal),
                 originNode: origin, destinationNode: goal,
                 distanceMeters: totalMeters, durationSeconds: totalSeconds,
                 roads: roads, polyline: polyline)
}

/// Optimal-then-greedy wrapper (mirrors `findRouteSpatialFiltered`,
/// highwayOnly=false): try the admissible search under a pop budget, fall
/// back to a greedy search on bail/skip so long routes still return.
func routeSpatial(graph: SpatialGraph, index: SZCIIndex,
                  origin: Int, goal: Int) async -> Route? {
    func coord(_ n: Int) -> (Double, Double) {
        (Double(index.nodesScaled[n * 2]) / 1e7, Double(index.nodesScaled[n * 2 + 1]) / 1e7)
    }
    let o = coord(origin), g = coord(goal)
    let crowKm = haversineMeters(o.0, o.1, g.0, g.1) / 1000
    if crowKm <= 800 {
        if let r = await aStarSpatial(graph: graph, index: index, origin: origin,
                                      goal: goal, greedyWeight: 1.0, popLimit: 200_000) {
            return r
        }
    }
    return await aStarSpatial(graph: graph, index: index, origin: origin,
                              goal: goal, greedyWeight: 1.5, popLimit: 400_000)
}

private func distSq(_ a: (lat: Double, lon: Double), _ b: (lat: Double, lon: Double)) -> Double {
    let dla = a.lat - b.lat
    let dlo = a.lon - b.lon
    return dla * dla + dlo * dlo
}

// MARK: - Tiny binary min-heap

private struct QueueItem: Comparable {
    let f: Double
    /// g at push time — lets the pop loop detect stale entries exactly
    /// (`item.g > gScore[node]`) without recomputing the heuristic.
    let g: Double
    let tiebreaker: Int
    let node: Int

    static func < (lhs: QueueItem, rhs: QueueItem) -> Bool {
        if lhs.f != rhs.f { return lhs.f < rhs.f }
        return lhs.tiebreaker < rhs.tiebreaker
    }
}

private struct MinHeap<T: Comparable> {
    private var items: [T] = []
    var isEmpty: Bool { items.isEmpty }

    mutating func push(_ x: T) {
        items.append(x)
        siftUp(items.count - 1)
    }

    mutating func pop() -> T? {
        guard !items.isEmpty else { return nil }
        items.swapAt(0, items.count - 1)
        let top = items.removeLast()
        if !items.isEmpty { siftDown(0) }
        return top
    }

    private mutating func siftUp(_ start: Int) {
        var i = start
        while i > 0 {
            let parent = (i - 1) / 2
            if items[i] < items[parent] {
                items.swapAt(i, parent)
                i = parent
            } else {
                return
            }
        }
    }

    private mutating func siftDown(_ start: Int) {
        var i = start
        let n = items.count
        while true {
            let l = 2 * i + 1
            let r = 2 * i + 2
            var smallest = i
            if l < n && items[l] < items[smallest] { smallest = l }
            if r < n && items[r] < items[smallest] { smallest = r }
            if smallest == i { return }
            items.swapAt(i, smallest)
            i = smallest
        }
    }
}
