import struct

import pytest

from mcpzim.routing import (
    Graph,
    astar,
    encode_graph_v2,
    haversine_m,
    _decode_varint,
    _zigzag_decode,
)
from mcpzim.routing import _varint, _zigzag_encode


# ---------------------------------------------------------------------------
# low-level codecs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, -1, 2, -2, 127, -127, 128, -128, 65535, -65535, 1_000_000, -1_000_000])
def test_zigzag_roundtrip(n):
    assert _zigzag_decode(_zigzag_encode(n)) == n


@pytest.mark.parametrize("n", [0, 1, 127, 128, 16384, 2_000_000_000])
def test_varint_roundtrip(n):
    enc = _varint(n)
    # Trailing byte must have the continuation bit cleared.
    assert enc[-1] & 0x80 == 0
    decoded, pos = _decode_varint(enc, 0)
    assert decoded == n
    assert pos == len(enc)


# ---------------------------------------------------------------------------
# graph parse / round-trip
# ---------------------------------------------------------------------------


def _grid_graph():
    """4-node unit-equator grid:
        B(0, 0.01) --- C(0.01, 0.01)
          |                |
        A(0, 0)    --- D(0.01, 0)
    Edge naming intentionally has A->B->C sharing one name and A->D->C
    sharing another, at different speeds, so we can check coalescing.
    """
    nodes = [(0.0, 0.0), (0.0, 0.01), (0.01, 0.01), (0.01, 0.0)]
    names = ["", "North Rd", "West Rd"]  # names[0] must be "".
    d = haversine_m(0.0, 0.0, 0.0, 0.01)
    edges = [
        (0, 1, d, 50, 1),
        (1, 2, d, 50, 1),
        (0, 3, d, 30, 2),
        (3, 2, d, 30, 2),
    ]
    return nodes, names, d, edges


def test_header_magic_and_counts():
    nodes, names, _, edges = _grid_graph()
    blob = encode_graph_v2(nodes=nodes, edges=edges, names=names)
    magic, version, numNodes, numEdges, numGeoms, _, numNames, _ = struct.unpack_from("<4s7I", blob, 0)
    assert magic == b"SZRG"
    assert version == 2
    assert numNodes == 4
    assert numEdges == 4
    assert numGeoms == 0
    assert numNames == 3


def test_graph_parse_roundtrip():
    nodes, names, d, edges = _grid_graph()
    blob = encode_graph_v2(nodes=nodes, edges=edges, names=names)
    g = Graph.parse(blob)
    assert g.num_nodes == 4
    assert g.num_edges == 4
    # Adjacency: node 0 has two outgoing edges, nodes 1 and 3 have one each.
    assert g.adj_offsets[1] - g.adj_offsets[0] == 2
    # Distance should match within sub-decimeter rounding.
    for e_dist in g.edge_dist_m:
        assert abs(e_dist - d) < 0.2
    # Speeds decoded intact.
    assert sorted(g.edge_speed_kmh) == [30, 30, 50, 50]
    # Names[1] through names[2] preserved.
    assert g.names[1:] == ["North Rd", "West Rd"]


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------


def test_astar_prefers_faster_route_and_coalesces_by_name():
    nodes, names, d, edges = _grid_graph()
    blob = encode_graph_v2(nodes=nodes, edges=edges, names=names)
    g = Graph.parse(blob)

    route = astar(g, 0, 2)  # A -> C
    assert route is not None

    # Both candidate routes are 2*d metres but one is 50 km/h and one is 30.
    # A* must pick the 50 km/h ("North Rd") path.
    assert [r.name for r in route.roads] == ["North Rd"]
    assert pytest.approx(route.distance_m, rel=1e-3) == 2 * d
    # 2*d m / (50/3.6) s = 2*d * 0.072 s ≈ 160 s.
    assert route.duration_s == pytest.approx(2 * d * 3.6 / 50, rel=1e-3)


def test_astar_returns_none_for_unreachable():
    # Two disjoint components: 0 <-> 1, 2 <-> 3.
    names = [""]
    d = haversine_m(0, 0, 0, 0.001)
    nodes = [(0.0, 0.0), (0.0, 0.001), (1.0, 1.0), (1.0, 1.001)]
    edges = [
        (0, 1, d, 50, 0),
        (1, 0, d, 50, 0),
        (2, 3, d, 50, 0),
        (3, 2, d, 50, 0),
    ]
    g = Graph.parse(encode_graph_v2(nodes=nodes, edges=edges, names=names))
    assert astar(g, 0, 2) is None


def test_astar_zero_length_origin_equals_goal():
    nodes, names, _, edges = _grid_graph()
    g = Graph.parse(encode_graph_v2(nodes=nodes, edges=edges, names=names))
    route = astar(g, 0, 0)
    assert route is not None
    assert route.distance_m == 0.0
    assert route.roads == []


def test_nearest_node_picks_closest_by_haversine():
    nodes, names, _, edges = _grid_graph()
    g = Graph.parse(encode_graph_v2(nodes=nodes, edges=edges, names=names))
    # A point right next to node C.
    assert g.nearest_node(0.0099, 0.0099) == 2
    assert g.nearest_node(0.0, 0.0) == 0


# ---------------------------------------------------------------------------
# geometry decode
# ---------------------------------------------------------------------------


def test_geom_decoder_roundtrip():
    # Build a 3-point polyline blob by hand using the same varint/zigzag codecs.
    pts = [(10.0, 20.0), (10.0001, 20.0002), (10.0003, 20.0001)]
    # First: lon0, lat0 as int32 absolute.
    prev_lat = int(round(pts[0][0] * 1e7))
    prev_lon = int(round(pts[0][1] * 1e7))
    buf = bytearray(struct.pack("<ii", prev_lon, prev_lat))
    for lat, lon in pts[1:]:
        lat_e7 = int(round(lat * 1e7))
        lon_e7 = int(round(lon * 1e7))
        buf.extend(_varint(_zigzag_encode(lon_e7 - prev_lon)))
        buf.extend(_varint(_zigzag_encode(lat_e7 - prev_lat)))
        prev_lat = lat_e7
        prev_lon = lon_e7

    decoded = Graph._decode_geom(bytes(buf), 0, len(buf))
    assert len(decoded) == 3
    for (got_lat, got_lon), (exp_lat, exp_lon) in zip(decoded, pts):
        assert abs(got_lat - exp_lat) < 1e-7
        assert abs(got_lon - exp_lon) < 1e-7
