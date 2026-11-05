"""NumPy reference assembly for Circulax; evaluation buffers remain raw-sized."""

import numpy as np


def allocate_raw_indices(topology, terminal_indices, next_unknown):
    """Allocate only surviving internals. Ground is -1, never an array index.

    Terminal nets must already be unioned across the circuit for any terminal
    shorts indicated by topology, including shorts to ground.
    """
    mapping = topology.raw_to_representative
    terminals = np.asarray(terminal_indices, dtype=np.int64)
    if terminals.shape != (len(mapping), topology.num_terminals):
        raise ValueError("Expected one global index per external terminal")
    indices = np.full(mapping.shape, -1, dtype=np.int64)
    for d, representatives in enumerate(mapping):
        global_reps = {-1: -1}
        for t in range(topology.num_terminals):
            rep = int(representatives[t])
            index = int(terminals[d, t])
            if rep in global_reps and global_reps[rep] != index:
                raise ValueError("Union shorted terminal nets before allocation")
            global_reps[rep] = index
        for rep in np.unique(representatives):
            rep = int(rep)
            if rep not in global_reps:
                global_reps[rep] = next_unknown
                next_unknown += 1
        indices[d] = [global_reps[int(rep)] for rep in representatives]
    return indices, next_unknown


def gather_voltages(y, raw_indices):
    """Append a literal zero so -1 never selects the final real unknown."""
    return np.concatenate([y, [0]])[np.where(raw_indices < 0, len(y), raw_indices)]


def scatter_stamps(topology, raw_indices, f, g, q, c, system_size):
    """Sum physical rows/columns into dense reference global F, G, Q, C.

    Production implementations can precompute COO row/column masks instead.
    """
    rows = topology.physical_row_mask & (raw_indices >= 0)
    cols = raw_indices >= 0
    matrix_mask = rows[:, :, None] & cols[:, None, :]
    rr = np.broadcast_to(raw_indices[:, :, None], matrix_mask.shape)[matrix_mask]
    cc = np.broadcast_to(raw_indices[:, None, :], matrix_mask.shape)[matrix_mask]
    results = []
    for stamp, matrix in [(f, False), (g, True), (q, False), (c, True)]:
        out = np.zeros(
            (system_size, system_size) if matrix else system_size,
            dtype=np.asarray(stamp).dtype,
        )
        if matrix:
            np.add.at(
                out, (rr, cc), np.asarray(stamp).reshape(matrix_mask.shape)[matrix_mask]
            )
        else:
            np.add.at(out, raw_indices[rows], np.asarray(stamp)[rows])
        results.append(out)
    return tuple(results)
