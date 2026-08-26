"""
The same Game-of-Life-equivalent network as network.py, but built as ONE
flat network over the whole grid, not many small per-cell inferences.

Layer 1 (input):  rows*cols        the entire grid, flattened row-major
Layer 2:          rows*cols*4      neurons 2.1..2.4 at every position
Layer 3:          rows*cols*3      neurons 3.1..3.3 at every position
Layer 4 (output): rows*cols        the entire next-epoch grid, flattened

Each layer is one big weighted-sum + step-activation matrix multiply over
its input vector -- there's no per-cell object, no loop that re-instantiates
a small network per position. The matrices are sparse (most entries are 0,
since each neuron still only depends on a handful of positions) but they
are real matrices, built once in __init__ and reused for every forward().

This file is separate from network.py on purpose: network.py's per-cell
ConwayNet (used by server.py / index.html) is untouched. The actual weight
numbers aren't re-derived here either -- they're pulled from network.py's
describe_network(), so there's still exactly one place those numbers live;
this file only contributes a different way of computing with them.
"""

from network import describe_network

MAX_DIM = 10

# Row-major 3x3 neighbor offsets; index 4 is the center. Matches the patch
# order describe_network()'s per-neuron weight lists are built against.
_NEIGHBOR_OFFSETS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]


def _flat_index(cols, i, j):
    return i * cols + j


class MatrixLayer:
    """weight_rows[k] is a sparse {input_index: weight} dict for output
    neuron k; biases[k] is its bias. forward(x) = step(W @ x + b), applied
    to the whole vector at once."""

    def __init__(self, weight_rows, biases):
        self.weight_rows = weight_rows
        self.biases = biases

    def forward(self, x):
        out = []
        for row, bias in zip(self.weight_rows, self.biases):
            total = bias
            for idx, w in row.items():
                total += w * x[idx]
            out.append(1 if total >= 0 else 0)
        return out


class FullGridNet:
    def __init__(self, rows, cols):
        if not (3 <= rows <= MAX_DIM and 3 <= cols <= MAX_DIM):
            raise ValueError(f"grid must be between 3x3 and {MAX_DIM}x{MAX_DIM}")
        self.rows = rows
        self.cols = cols
        spec = describe_network()
        self.layer2 = self._build_layer2(spec)
        self.layer3 = self._build_layer3(spec)
        self.layer4 = self._build_layer4(spec)

    def _build_layer2(self, spec):
        # Output order per position: 2.1, 2.2, 2.3, 2.4 (4 rows per cell).
        neuron_specs = [spec["layer2"]["n1"], spec["layer2"]["n2"], spec["layer2"]["n3"], spec["layer2"]["n4"]]
        weight_rows, biases = [], []
        for i in range(self.rows):
            for j in range(self.cols):
                for neuron in neuron_specs:
                    row = {}
                    for k, (di, dj) in enumerate(_NEIGHBOR_OFFSETS):
                        w = neuron["weights"][k]
                        ni, nj = i + di, j + dj
                        if w != 0 and 0 <= ni < self.rows and 0 <= nj < self.cols:
                            row[_flat_index(self.cols, ni, nj)] = w
                    weight_rows.append(row)
                    biases.append(neuron["bias"])
        return MatrixLayer(weight_rows, biases)

    def _build_layer3(self, spec):
        # Each layer-3 neuron at position p only reads p's own 4 layer-2
        # outputs (laid out consecutively: p*4 .. p*4+3). Output order per
        # position: 3.1, 3.2, 3.3 (3 rows per cell).
        neuron_specs = [spec["layer3"]["n1"], spec["layer3"]["n2"], spec["layer3"]["n3"]]
        weight_rows, biases = [], []
        for i in range(self.rows):
            for j in range(self.cols):
                base = _flat_index(self.cols, i, j) * 4
                for neuron in neuron_specs:
                    row = {base + k: w for k, w in enumerate(neuron["weights"]) if w != 0}
                    weight_rows.append(row)
                    biases.append(neuron["bias"])
        return MatrixLayer(weight_rows, biases)

    def _build_layer4(self, spec):
        # Each output neuron at position p only reads p's own 3 layer-3
        # outputs (laid out consecutively: p*3 .. p*3+2). One row per cell.
        out_spec = spec["output"]
        weight_rows, biases = [], []
        for i in range(self.rows):
            for j in range(self.cols):
                base = _flat_index(self.cols, i, j) * 3
                row = {base + k: w for k, w in enumerate(out_spec["weights"]) if w != 0}
                weight_rows.append(row)
                biases.append(out_spec["bias"])
        return MatrixLayer(weight_rows, biases)

    def forward(self, grid):
        rows, cols = len(grid), len(grid[0])
        if rows != self.rows or cols != self.cols:
            raise ValueError(f"expected a {self.rows}x{self.cols} grid, got {rows}x{cols}")

        flat_input = [grid[i][j] for i in range(rows) for j in range(cols)]
        flat_l2 = self.layer2.forward(flat_input)
        flat_l3 = self.layer3.forward(flat_l2)
        flat_l4 = self.layer4.forward(flat_l3)

        output_grid = [flat_l4[i * cols:(i + 1) * cols] for i in range(rows)]
        return flat_l2, flat_l3, output_grid


def render(grid, on="#", off="."):
    return "\n".join("".join(on if c else off for c in row) for row in grid)


def _lcg_grid(rows, cols, seed=12345):
    """Same tiny PRNG as network.py's demo, kept import-free."""
    state = seed
    grid = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            state = (1103515245 * state + 12345) % 2147483648
            row.append(1 if (state % 3 == 0) else 0)
        grid.append(row)
    return grid


if __name__ == "__main__":
    grid = _lcg_grid(10, 10)
    net = FullGridNet(10, 10)

    print("Input layer (10x10):")
    print(render(grid))

    _, _, output = net.forward(grid)
    print(f"\nOutput layer (next epoch) ({len(output)}x{len(output[0])}):")
    print(render(output))

    print(f"\nLayer sizes: input={net.rows * net.cols}, "
          f"layer2={len(net.layer2.weight_rows)}, "
          f"layer3={len(net.layer3.weight_rows)}, "
          f"output={len(net.layer4.weight_rows)}")

    # Cross-check against network.py's per-cell ConwayNet, to prove this
    # flat-matrix formulation computes the exact same function.
    from network import ConwayNet
    import random

    random.seed(7)
    mismatches = 0
    for trial in range(20):
        rows_t, cols_t = random.randint(3, 10), random.randint(3, 10)
        g = [[random.choice([0, 1]) for _ in range(cols_t)] for _ in range(rows_t)]
        _, _, flat_out = FullGridNet(rows_t, cols_t).forward(g)
        _, _, cell_out = ConwayNet(g).forward()
        if flat_out != cell_out:
            mismatches += 1
            print(f"MISMATCH trial={trial} dims={rows_t}x{cols_t}")
    print(f"\nCross-check vs ConwayNet: {20 - mismatches}/20 grids matched exactly.")
