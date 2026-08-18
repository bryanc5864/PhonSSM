"""
Anatomical graph for the 27-joint SAM-SLR / HRNet whole-body skeleton, with an
explicit part decomposition used by PhonoGraph.

The 27 joints decompose cleanly (verified empirically from the edge list and from
per-joint motion energy):
  body  = indices 0..6   (nose, L/R shoulder, L/R elbow, L/R wrist)  -- low motion
  handA = indices 7..16  (10 joints: palm base + fingers)            -- high motion
  handB = indices 17..26 (10 joints)                                 -- high motion

We build, per part, a normalized adjacency (self + symmetric neighbor) restricted
to that part's own joints, plus the two "attachment" joints that connect a hand to
its wrist so the hand encoder still sees where the hand is rooted.

Edge list is the same INWARD_ORI_INDEX (shifted by -5 to 0-based) as the faithful
DSTA graph, so we operate on identical skeleton topology.
"""
import numpy as np

BODY = list(range(0, 7))      # 7 joints
HAND_A = list(range(7, 17))   # 10 joints
HAND_B = list(range(17, 27))  # 10 joints

# 0-based edges (INWARD_ORI_INDEX shifted by -5)
INWARD = [
    (0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6),        # body
    (7, 8), (7, 9), (7, 11), (7, 13), (7, 15),             # handA palm->finger-bases
    (9, 10), (11, 12), (13, 14), (15, 16),                 # handA finger tips
    (17, 18), (17, 19), (17, 21), (17, 23), (17, 25),      # handB
    (19, 20), (21, 22), (23, 24), (25, 26),                # handB tips
    (5, 7), (6, 17),                                        # wrist->hand attachments
]

# wrist that each hand attaches to (body joint index)
HAND_A_WRIST = 5
HAND_B_WRIST = 6


def _norm_adj(edges, nodes):
    """Symmetric normalized adjacency (D^-1/2 (A+I) D^-1/2) over the given node set,
    returned as a (len(nodes), len(nodes)) matrix with local indexing."""
    idx = {n: i for i, n in enumerate(nodes)}
    k = len(nodes)
    A = np.eye(k, dtype=np.float32)
    for a, b in edges:
        if a in idx and b in idx:
            A[idx[a], idx[b]] = 1.0
            A[idx[b], idx[a]] = 1.0
    D = A.sum(-1)
    Dinv = np.diag((D ** -0.5).astype(np.float32))
    return (Dinv @ A @ Dinv).astype(np.float32)


def part_nodes():
    """Return the joint-index lists for each part encoder. Each hand includes its
    attachment wrist so the hand knows its body-relative root."""
    a = [HAND_A_WRIST] + HAND_A       # 11 joints
    b = [HAND_B_WRIST] + HAND_B       # 11 joints
    return {'body': BODY, 'handA': a, 'handB': b}


def part_adjacency():
    """Normalized within-part adjacency matrices (local indexing)."""
    parts = part_nodes()
    return {name: _norm_adj(INWARD, nodes) for name, nodes in parts.items()}


# full 27-node normalized adjacency (for any full-graph variant / fallback)
def full_adjacency():
    return _norm_adj(INWARD, list(range(27)))
