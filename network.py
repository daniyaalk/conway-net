"""
A neural network (built from scratch, no libraries) that decides whether
each cell of a 10x10-max grid is on in the next epoch. Strictly layered --
every layer depends only on the outputs of the layer before it -- and every
neuron is a single weighted-sum + step-activation unit.

Layer 1 (receptive field): for each cell, its 3x3 neighborhood (itself plus
    8 neighbors), row-major flattened, index 4 = itself. Out-of-bounds
    neighbors are 0.

Layer 2 (4 neurons), each wired to layer 1:
    2.1 fires if 2+ of the 8 neighbors are on
    2.2 fires if 3+ of the 8 neighbors are on
    2.3 fires if 4+ of the 8 neighbors are on
    2.4 fires if the same cell (the center) is NOT on

Layer 3 (3 neurons), each wired to layer 2's (2.1, 2.2, 2.3, 2.4):
    3.1 fires if 2.3 is firing                          (overpopulation)
    3.2 fires if 2.4 and 2.2 and not 2.3 are firing      (birth)
    3.3 fires if 2.1 is firing and 2.4 is not firing     (continuation)

Layer 4 (output), wired to layer 3's (3.1, 3.2, 3.3):
    fires if 3.1 is not firing and at least one of 3.2, 3.3 is firing

This reproduces Conway's Game of Life (B3/S23) exactly: verified by
enumerating every (center, neighbor-count) combination against the real
rule -- see the network-vs-GoL check in the project's test notes.
"""

MAX_DIM = 10


class Neuron:
    """A weighted-sum + step-activation unit: fires (1) if sum(w*x) + bias >= 0."""

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def fire(self, inputs):
        total = self.bias
        for w, x in zip(self.weights, inputs):
            total += w * x
        return 1 if total >= 0 else 0


def make_neighbor_neuron(threshold):
    # Layer 2 neurons 2.1/2.2/2.3: wired to the 8 neighbors of the 3x3
    # patch (index 4, the center, has weight 0); fires if `threshold` or
    # more of them are on.
    weights = [1, 1, 1, 1, 0, 1, 1, 1, 1]
    return Neuron(weights, bias=-threshold)


def make_center_off_neuron():
    # Layer 2 neuron 2.4: wired only to the center (index 4); fires iff
    # that same cell is NOT on.
    weights = [0, 0, 0, 0, -1, 0, 0, 0, 0]
    return Neuron(weights, bias=0)


def make_layer3_neurons():
    # Each takes layer 2's (n2.1, n2.2, n2.3, n2.4).
    n3_1 = Neuron(weights=[0, 0, 1, 0], bias=-1)   # fires if 2.3 firing
    n3_2 = Neuron(weights=[0, 1, -1, 1], bias=-2)  # birth: 2.4 & 2.2 & not 2.3
    n3_3 = Neuron(weights=[2, 0, 0, -1], bias=-2)  # continuation: 2.1 & not 2.4
    return n3_1, n3_2, n3_3


def make_output_neuron():
    # Layer 4: takes layer 3's (n3.1, n3.2, n3.3).
    # Fires iff 3.1 is off and (3.2 or 3.3) -- the -2 weight on 3.1 acts as
    # a veto strong enough to override 3.2 + 3.3 firing together.
    return Neuron(weights=[-2, 1, 1], bias=-1)


class ConwayNet:
    def __init__(self, grid):
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        if not (3 <= rows <= MAX_DIM and 3 <= cols <= MAX_DIM):
            raise ValueError(f"grid must be between 3x3 and {MAX_DIM}x{MAX_DIM}")
        if any(len(row) != cols for row in grid):
            raise ValueError("grid rows must all be the same length")
        if any(cell not in (0, 1) for row in grid for cell in row):
            raise ValueError("grid cells must be 0 or 1")

        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.n2_1 = make_neighbor_neuron(threshold=2)
        self.n2_2 = make_neighbor_neuron(threshold=3)
        self.n2_3 = make_neighbor_neuron(threshold=4)
        self.n2_4 = make_center_off_neuron()
        self.n3_1, self.n3_2, self.n3_3 = make_layer3_neurons()
        self.output_neuron = make_output_neuron()

    def _patch(self, i, j):
        # 3x3 patch centered at (i, j), row-major, index 4 = center itself;
        # out-of-bounds neighbors are 0.
        patch = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    patch.append(self.grid[ni][nj])
                else:
                    patch.append(0)
        return patch

    def forward(self):
        # Layer 2: rows x cols x 4
        layer2 = [[None] * self.cols for _ in range(self.rows)]
        # Layer 3: rows x cols x 3
        layer3 = [[None] * self.cols for _ in range(self.rows)]
        # Layer 4 (output): rows x cols
        layer4 = [[0] * self.cols for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                patch = self._patch(i, j)
                n21 = self.n2_1.fire(patch)
                n22 = self.n2_2.fire(patch)
                n23 = self.n2_3.fire(patch)
                n24 = self.n2_4.fire(patch)
                layer2[i][j] = (n21, n22, n23, n24)

                l2 = layer2[i][j]
                n31 = self.n3_1.fire(l2)
                n32 = self.n3_2.fire(l2)
                n33 = self.n3_3.fire(l2)
                layer3[i][j] = (n31, n32, n33)

                layer4[i][j] = self.output_neuron.fire(layer3[i][j])

        return layer2, layer3, layer4


def run(grid, epochs):
    """Advance `grid` forward `epochs` full epochs (each epoch = one
    ConwayNet.forward() pass, output fed back in as the next input).
    Returns the grid history from epoch 0 (the input) through `epochs`,
    so history[-1] is the final grid and history[-2] is what fed it."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    history = [grid]
    current = grid
    for _ in range(epochs):
        _, _, current = ConwayNet(current).forward()
        history.append(current)
    return history


def describe_network():
    """The network's actual weights/biases, for visualizing it -- the only
    place these numbers live, so a UI renders the real graph instead of a
    hardcoded second copy of it."""
    n2_1 = make_neighbor_neuron(threshold=2)
    n2_2 = make_neighbor_neuron(threshold=3)
    n2_3 = make_neighbor_neuron(threshold=4)
    n2_4 = make_center_off_neuron()
    n3_1, n3_2, n3_3 = make_layer3_neurons()
    output = make_output_neuron()

    def spec(neuron):
        return {"weights": neuron.weights, "bias": neuron.bias}

    return {
        "layer2": {"n1": spec(n2_1), "n2": spec(n2_2), "n3": spec(n2_3), "n4": spec(n2_4)},
        "layer3": {"n1": spec(n3_1), "n2": spec(n3_2), "n3": spec(n3_3)},
        "output": spec(output),
    }


def render(grid, on="#", off="."):
    return "\n".join("".join(on if c else off for c in row) for row in grid)


def _lcg_grid(rows, cols, seed=12345):
    """Tiny linear-congruential PRNG so the demo needs no imports at all."""
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

    print("Input layer (10x10):")
    print(render(grid))

    net = ConwayNet(grid)
    layer2, layer3, layer4 = net.forward()

    for idx, label in enumerate(["2.1 (>=2 neighbors)", "2.2 (>=3 neighbors)", "2.3 (>=4 neighbors)", "2.4 (center off)"]):
        m = [[cell[idx] for cell in row] for row in layer2]
        print(f"\nLayer 2, neuron {label}:")
        print(render(m))

    for idx, label in enumerate(["3.1 (overpopulation)", "3.2 (birth)", "3.3 (continuation)"]):
        m = [[cell[idx] for cell in row] for row in layer3]
        print(f"\nLayer 3, neuron {label}:")
        print(render(m))

    print(f"\nOutput layer (next epoch) ({len(layer4)}x{len(layer4[0])}):")
    print(render(layer4))
