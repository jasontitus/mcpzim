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
/// - Admissible heuristic assumes no road is faster than 100 km/h, giving
///   `haversine / (100/3.6)`.
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
    let speedCeilingMps = 100.0 / 3.6

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
    open.push(QueueItem(f: heuristic(origin), tiebreaker: counter, node: origin))
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
        if curG < current.f - heuristic(current.node) - 1e-9 { continue }

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
                open.push(QueueItem(f: tentative + heuristic(target), tiebreaker: counter, node: target))
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

private func distSq(_ a: (lat: Double, lon: Double), _ b: (lat: Double, lon: Double)) -> Double {
    let dla = a.lat - b.lat
    let dlo = a.lon - b.lon
    return dla * dla + dlo * dlo
}

// MARK: - Tiny binary min-heap

private struct QueueItem: Comparable {
    let f: Double
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
