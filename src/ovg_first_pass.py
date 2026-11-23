"""First-pass edge decoding shared by training and evaluation."""


def boundary_candidate(alpha, depth_gate, epsilon):
    return (alpha < 1-epsilon) & depth_gate


def observed_split(alpha, depth_gate, quality, epsilon):
    """A boundary can split only when a background candidate was returned."""
    return boundary_candidate(alpha, depth_gate, epsilon) & (quality > 0)
