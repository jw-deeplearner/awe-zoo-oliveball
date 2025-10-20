
def assert_contiguous_labels_index(labels_index: dict):
    """Require labels_index values to be exactly 0..N-1 (no gaps, no negatives)."""
    indices = list(labels_index.values())
    n = len(indices)
    if set(indices) != set(range(n)):
        raise ValueError(f"labels_index must be contiguous 0..{n-1}, got values={sorted(indices)}")