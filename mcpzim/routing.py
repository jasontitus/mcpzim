"""StreetZim routing: parse SZRG v2 binary graphs and run A* shortest paths.

The binary layout mirrors streetzim's ``routing-data/graph.bin`` as produced by
``create_osm_zim.py``; the A* cost and heuristic match the JavaScript viewer so
results are identical to what the map UI would return.
"""

from __future__ import annotations

import heapq
import logging
import math
import struct
import threading
from dataclasses import dataclass
from typing import Sequence

from .library import OpenZim

log = logging.getLogger(__name__)


EARTH_R_M = 6_371_000.0
_SPEED_CEILING_KMH = 100.0  # heuristic assumes no edge is faster than this.
_NO_GEOM = 0xFFFFFF


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R_M * math.asin(math.sqrt(a))


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    result = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift >= 64:
            raise ValueError("varint too long")


def _zigzag_decode(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


@dataclass
class Graph:
    """Parsed SZRG v2 routing graph, kept entirely in memory."""

    num_nodes: int
    num_edges: int
    # Parallel arrays so we can use tight Python loops / array lookups.
    lat: list[float]            # degrees
    lon: list[float]            # degrees
    adj_offsets: list[int]      # len == num_nodes + 1
    edge_targets: list[int]
    edge_dist_m: list[float]
    edge_speed_kmh: list[int]
    edge_geom_idx: list[int]    # -1 for "no polyline"
    edge_name_idx: list[int]    # 0 for unnamed
    names: list[str]            # names[0] == "" (sentinel)
    geoms: list[list[tuple[float, float]]]  # polyline as (lat, lon) pairs

    @classmethod
    def parse(cls, blob: bytes) -> "Graph":
        if len(blob) < 32:
            raise ValueError("graph.bin too small")
        magic, version, num_nodes, num_edges, num_geoms, geom_bytes, num_names, names_bytes = (
            struct.unpack_from("<4s7I", blob, 0)
        )
        if magic != b"SZRG":
            raise ValueError(f"bad magic {magic!r}")
        if version != 2:
            raise ValueError(f"unsupported SZRG version {version}")

        pos = 32

        nodes = struct.unpack_from(f"<{2 * num_nodes}i", blob, pos)
        pos += 8 * num_nodes
        # Node array is [lat_e7, lon_e7] pairs.
        lat = [nodes[2 * i] / 1e7 for i in range(num_nodes)]
        lon = [nodes[2 * i + 1] / 1e7 for i in range(num_nodes)]

        adj_offsets = list(struct.unpack_from(f"<{num_nodes + 1}I", blob, pos))
        pos += 4 * (num_nodes + 1)

        edges = struct.unpack_from(f"<{4 * num_edges}I", blob, pos)
        pos += 16 * num_edges
        edge_targets = [edges[4 * i] for i in range(num_edges)]
        edge_dist_m = [edges[4 * i + 1] / 10.0 for i in range(num_edges)]
        edge_speed_kmh = [(edges[4 * i + 2] >> 24) & 0xFF for i in range(num_edges)]
        edge_geom_idx = [
            (edges[4 * i + 2] & 0x00FFFFFF)
            if (edges[4 * i + 2] & 0x00FFFFFF) != _NO_GEOM
            else -1
            for i in range(num_edges)
        ]
        edge_name_idx = [edges[4 * i + 3] for i in range(num_edges)]

        geom_offsets = list(struct.unpack_from(f"<{num_geoms + 1}I", blob, pos))
        pos += 4 * (num_geoms + 1)
        geom_blob = blob[pos : pos + geom_bytes]
        pos += geom_bytes

        name_offsets = list(struct.unpack_from(f"<{num_names + 1}I", blob, pos))
        pos += 4 * (num_names + 1)
        names_blob = blob[pos : pos + names_bytes]
        pos += names_bytes

        geoms = [cls._decode_geom(geom_blob, geom_offsets[i], geom_offsets[i + 1]) for i in range(num_geoms)]

        names: list[str] = []
        for i in range(num_names):
            s = names_blob[name_offsets[i] : name_offsets[i + 1]]
            try:
                names.append(s.decode("utf-8"))
            except UnicodeDecodeError:
                names.append(s.decode("utf-8", "replace"))

        return cls(
            num_nodes=num_nodes,
            num_edges=num_edges,
            lat=lat,
            lon=lon,
            adj_offsets=adj_offsets,
            edge_targets=edge_targets,
            edge_dist_m=edge_dist_m,
            edge_speed_kmh=edge_speed_kmh,
            edge_geom_idx=edge_geom_idx,
            edge_name_idx=edge_name_idx,
            names=names,
            geoms=geoms,
        )

    @staticmethod
    def _decode_geom(blob: bytes, start: int, end: int) -> list[tuple[float, float]]:
        if end <= start:
            return []
        lon0_e7, lat0_e7 = struct.unpack_from("<ii", blob, start)
        pos = start + 8
        pts: list[tuple[float, float]] = [(lat0_e7 / 1e7, lon0_e7 / 1e7)]
        lat_e7 = lat0_e7
        lon_e7 = lon0_e7
        while pos < end:
            raw_dlon, pos = _decode_varint(blob, pos)
            raw_dlat, pos = _decode_varint(blob, pos)
            lon_e7 += _zigzag_decode(raw_dlon)
            lat_e7 += _zigzag_decode(raw_dlat)
            pts.append((lat_e7 / 1e7, lon_e7 / 1e7))
        return pts

    def name(self, idx: int) -> str:
        if idx <= 0 or idx >= len(self.names) + 1:
            return ""
        # ``names`` is 0-indexed but idx==0 is the unnamed sentinel in the on-disk
        # name offset table — the JS reader returns "" for 0 and we match that.
        if idx >= len(self.names):
            return ""
        return self.names[idx]

    def nearest_node(self, lat: float, lon: float) -> int:
        """Linear scan for the closest node by haversine distance."""
        # Haversine everywhere — cheap enough for the scale of city/state graphs.
        best_i = -1
        best_d = math.inf
        for i in range(self.num_nodes):
            d = haversine_m(lat, lon, self.lat[i], self.lon[i])
            if d < best_d:
                best_d = d
                best_i = i
        return best_i


@dataclass
class RoadSegment:
    """A run of consecutive edges sharing a single street name."""

    name: str
    distance_m: float
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "name": self.name or "(unnamed road)",
            "distance_m": round(self.distance_m, 1),
            "duration_s": round(self.duration_s, 1),
        }


@dataclass
class Route:
    origin: tuple[float, float]
    destination: tuple[float, float]
    origin_node: int
    destination_node: int
    distance_m: float
    duration_s: float
    roads: list[RoadSegment]
    polyline: list[tuple[float, float]]  # (lat, lon) vertices

    def to_dict(self) -> dict:
        return {
            "origin": {"lat": self.origin[0], "lon": self.origin[1]},
            "destination": {"lat": self.destination[0], "lon": self.destination[1]},
            "origin_node": self.origin_node,
            "destination_node": self.destination_node,
            "distance_m": round(self.distance_m, 1),
            "distance_km": round(self.distance_m / 1000.0, 3),
            "duration_s": round(self.duration_s, 1),
            "duration_min": round(self.duration_s / 60.0, 1),
            "roads": [r.to_dict() for r in self.roads],
            "polyline": [[round(lat, 7), round(lon, 7)] for lat, lon in self.polyline],
            "turn_by_turn": [
                f"{r.name or '(unnamed road)'} for {r.distance_m / 1000:.2f} km"
                f" (~{r.duration_s / 60:.1f} min)"
                for r in self.roads
            ],
        }


def astar(graph: Graph, origin: int, goal: int) -> Route | None:
    """Shortest travel-time path using A* with a haversine / 100 km/h heuristic."""
    if origin == goal:
        return Route(
            origin=(graph.lat[origin], graph.lon[origin]),
            destination=(graph.lat[goal], graph.lon[goal]),
            origin_node=origin,
            destination_node=goal,
            distance_m=0.0,
            duration_s=0.0,
            roads=[],
            polyline=[(graph.lat[origin], graph.lon[origin])],
        )

    goal_lat = graph.lat[goal]
    goal_lon = graph.lon[goal]

    def h(node: int) -> float:
        return haversine_m(graph.lat[node], graph.lon[node], goal_lat, goal_lon) / (
            _SPEED_CEILING_KMH / 3.6
        )

    came_from: dict[int, tuple[int, int]] = {}  # node -> (prev, edge_idx)
    g_score: dict[int, float] = {origin: 0.0}
    open_heap: list[tuple[float, int, int]] = []
    heapq.heappush(open_heap, (h(origin), 0, origin))
    counter = 1

    adj = graph.adj_offsets
    targets = graph.edge_targets
    dists = graph.edge_dist_m
    speeds = graph.edge_speed_kmh

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct_route(graph, origin, goal, came_from)
        cur_g = g_score[current]
        for e in range(adj[current], adj[current + 1]):
            speed = speeds[e] or 1  # guard — never divide by zero.
            cost = dists[e] * 3.6 / speed
            neigh = targets[e]
            tentative = cur_g + cost
            if tentative < g_score.get(neigh, math.inf):
                g_score[neigh] = tentative
                came_from[neigh] = (current, e)
                heapq.heappush(open_heap, (tentative + h(neigh), counter, neigh))
                counter += 1
    return None


def _reconstruct_route(
    graph: Graph,
    origin: int,
    goal: int,
    came_from: dict[int, tuple[int, int]],
) -> Route:
    # Walk back: collect (prev_node, edge_idx, this_node) in reverse.
    rev: list[tuple[int, int, int]] = []
    node = goal
    while node != origin:
        prev, edge = came_from[node]
        rev.append((prev, edge, node))
        node = prev
    rev.reverse()

    polyline: list[tuple[float, float]] = [(graph.lat[origin], graph.lon[origin])]
    roads: list[RoadSegment] = []
    total_m = 0.0
    total_s = 0.0

    for prev, edge, this in rev:
        dist_m = graph.edge_dist_m[edge]
        speed = graph.edge_speed_kmh[edge] or 1
        dur_s = dist_m * 3.6 / speed
        total_m += dist_m
        total_s += dur_s

        gi = graph.edge_geom_idx[edge]
        if gi >= 0 and gi < len(graph.geoms) and graph.geoms[gi]:
            pts = graph.geoms[gi]
            # Polylines are stored in forward-edge direction; reverse if this edge
            # goes the other way (endpoint test on the first/last point).
            if _dist2(pts[0], (graph.lat[prev], graph.lon[prev])) > _dist2(
                pts[-1], (graph.lat[prev], graph.lon[prev])
            ):
                pts = list(reversed(pts))
            polyline.extend(pts[1:])
        else:
            polyline.append((graph.lat[this], graph.lon[this]))

        name = graph.name(graph.edge_name_idx[edge])
        if roads and roads[-1].name == name:
            roads[-1].distance_m += dist_m
            roads[-1].duration_s += dur_s
        else:
            roads.append(RoadSegment(name=name, distance_m=dist_m, duration_s=dur_s))

    return Route(
        origin=(graph.lat[origin], graph.lon[origin]),
        destination=(graph.lat[goal], graph.lon[goal]),
        origin_node=origin,
        destination_node=goal,
        distance_m=total_m,
        duration_s=total_s,
        roads=roads,
        polyline=polyline,
    )


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class RouterCache:
    """Lazily decodes and caches the Graph for each streetzim ZIM."""

    def __init__(self) -> None:
        self._graphs: dict[str, Graph] = {}
        self._lock = threading.Lock()

    def graph_for(self, zim: OpenZim) -> Graph:
        key = str(zim.path.resolve())
        with self._lock:
            g = self._graphs.get(key)
            if g is not None:
                return g
        blob = _read_graph_bin(zim)
        g = Graph.parse(blob)
        with self._lock:
            self._graphs.setdefault(key, g)
        log.info("loaded routing graph from %s: %d nodes, %d edges", zim.path.name, g.num_nodes, g.num_edges)
        return g


def _read_graph_bin(zim: OpenZim) -> bytes:
    with zim.lock:
        entry = zim.archive.get_entry_by_path("routing-data/graph.bin")
        if getattr(entry, "is_redirect", False):
            entry = entry.get_redirect_entry()
        item = entry.get_item()
        return bytes(item.content)


def plan_route(
    zim: OpenZim,
    graph: Graph,
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> Route | None:
    origin = graph.nearest_node(origin_lat, origin_lon)
    goal = graph.nearest_node(dest_lat, dest_lon)
    if origin < 0 or goal < 0:
        return None
    route = astar(graph, origin, goal)
    if route is None:
        return None
    # Swap in the user's actual request coords so the response reflects them.
    route.origin = (origin_lat, origin_lon)
    route.destination = (dest_lat, dest_lon)
    return route


__all__ = [
    "Graph",
    "Route",
    "RoadSegment",
    "RouterCache",
    "astar",
    "haversine_m",
    "plan_route",
]


# -- helpers exposed for tests ------------------------------------------------

def encode_graph_v2(
    *,
    nodes: Sequence[tuple[float, float]],
    edges: Sequence[tuple[int, int, float, int, int]],
    names: Sequence[str],
    geoms: Sequence[Sequence[tuple[float, float]]] = (),
) -> bytes:
    """Produce a minimal SZRG v2 blob for tests.

    ``edges`` items are (src, dst, dist_m, speed_kmh, name_idx) — geometry indexing
    is not supported here; edges written by this helper always carry the ``no-geom``
    sentinel. ``names`` must have ``""`` at index 0.
    """
    if names and names[0] != "":
        raise ValueError("names[0] must be '' (unnamed sentinel)")

    num_nodes = len(nodes)
    num_edges = len(edges)
    num_geoms = len(geoms)
    num_names = len(names)

    # Build CSR adjacency.
    bucket: list[list[tuple[int, int, int, int]]] = [[] for _ in range(num_nodes)]
    for src, dst, dist_m, speed_kmh, name_idx in edges:
        bucket[src].append((dst, int(round(dist_m * 10)), speed_kmh, name_idx))
    adj_offsets = [0]
    flat_edges: list[tuple[int, int, int, int]] = []
    for b in bucket:
        flat_edges.extend(b)
        adj_offsets.append(len(flat_edges))

    # Geom blob.
    geom_offsets = [0]
    geom_parts: list[bytes] = []
    for g in geoms:
        if not g:
            geom_offsets.append(geom_offsets[-1])
            continue
        lat0, lon0 = g[0]
        buf = bytearray(struct.pack("<ii", int(round(lon0 * 1e7)), int(round(lat0 * 1e7))))
        prev_lat_e7 = int(round(lat0 * 1e7))
        prev_lon_e7 = int(round(lon0 * 1e7))
        for lat, lon in g[1:]:
            lat_e7 = int(round(lat * 1e7))
            lon_e7 = int(round(lon * 1e7))
            buf.extend(_varint(_zigzag_encode(lon_e7 - prev_lon_e7)))
            buf.extend(_varint(_zigzag_encode(lat_e7 - prev_lat_e7)))
            prev_lat_e7 = lat_e7
            prev_lon_e7 = lon_e7
        geom_parts.append(bytes(buf))
        geom_offsets.append(geom_offsets[-1] + len(buf))
    geom_blob = b"".join(geom_parts)
    # Pad to multiple of 4.
    pad = (-len(geom_blob)) & 3
    geom_blob += b"\x00" * pad
    geom_bytes = len(geom_blob)

    # Names blob.
    name_offsets = [0]
    name_parts: list[bytes] = []
    for s in names:
        enc = s.encode("utf-8")
        name_parts.append(enc)
        name_offsets.append(name_offsets[-1] + len(enc))
    names_blob = b"".join(name_parts)
    names_bytes = len(names_blob)

    out = bytearray()
    out.extend(struct.pack("<4s7I", b"SZRG", 2, num_nodes, num_edges, num_geoms, geom_bytes, num_names, names_bytes))
    for lat, lon in nodes:
        out.extend(struct.pack("<ii", int(round(lat * 1e7)), int(round(lon * 1e7))))
    out.extend(struct.pack(f"<{num_nodes + 1}I", *adj_offsets))
    for dst, dist_dm, speed_kmh, name_idx in flat_edges:
        speed_geom = ((speed_kmh & 0xFF) << 24) | _NO_GEOM
        out.extend(struct.pack("<IIII", dst, dist_dm, speed_geom, name_idx))
    out.extend(struct.pack(f"<{num_geoms + 1}I", *geom_offsets))
    out.extend(geom_blob)
    out.extend(struct.pack(f"<{num_names + 1}I", *name_offsets))
    out.extend(names_blob)
    return bytes(out)


def _zigzag_encode(n: int) -> int:
    return (n << 1) ^ (n >> 31) if n >= 0 else (n << 1) ^ (n >> 63)


def _varint(n: int) -> bytes:
    if n < 0:
        raise ValueError("varint expects non-negative")
    buf = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return bytes(buf)
