"""
The same Conway's Game of Life network as network.py, but with layer 3
eliminated. One extra weighted-sum + step neuron combines layer 2's four
outputs directly into the final answer, with no intermediate combining
layer in between. This is a lossless simplification, not an approximation:
it reproduces network.py's ConwayNet exactly, cell for cell and epoch for
epoch (verified below by exhaustive enumeration over all 512 possible 3x3
patches).

Layer 2 is untouched -- imported straight from network.py, not re-derived
-- so 2.1/2.2/2.3/2.4 fire exactly as they do there. What's new is a
single neuron that replaces layer 3 (3 neurons) and the old output neuron
(1 neuron) -- four neurons, 19 parameters -- with one:

    output fires iff  1*2.1 + 1*2.2 - 2*2.3 - 1*2.4 - 1  >= 0

The -2 weight on 2.3 (fires on 4+ neighbors) acts as a veto strong enough
to zero out the output on overpopulation regardless of anything else. The
-1 weight on 2.4 (fires when the center is off) exactly cancels 2.1's +1
contribution when the center is dead, so a dead cell needs both 2.1 AND
2.2 (i.e. exactly 3 neighbors) to clear the same threshold a live cell
clears with 2.1 alone (i.e. 2 or 3 neighbors) -- precisely the
survive-on-2-or-3 vs. birth-on-3 distinction the original layer 3 existed
to compute.
"""

from network import MAX_DIM, Neuron, make_center_off_neuron, make_neighbor_neuron


def make_reduced_output_neuron():
    # Takes layer 2's (2.1, 2.2, 2.3, 2.4) directly -- no layer 3 in between.
    return Neuron(weights=[1, 1, -2, -1], bias=-1)


class ReducedConwayNet:
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
        self.output_neuron = make_reduced_output_neuron()

    def _patch(self, i, j):
        # Same 3x3-patch convention as ConwayNet: row-major, index 4 =
        # center, out-of-bounds neighbors are 0.
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
        # Output: rows x cols
        output_grid = [[0] * self.cols for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                patch = self._patch(i, j)
                n21 = self.n2_1.fire(patch)
                n22 = self.n2_2.fire(patch)
                n23 = self.n2_3.fire(patch)
                n24 = self.n2_4.fire(patch)
                layer2[i][j] = (n21, n22, n23, n24)
                output_grid[i][j] = self.output_neuron.fire(layer2[i][j])

        return layer2, output_grid


def run(grid, epochs):
    """Same interface as network.run(): advances `grid` forward `epochs`
    epochs, returning the grid history from epoch 0 through `epochs`."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    history = [grid]
    current = grid
    for _ in range(epochs):
        _, current = ReducedConwayNet(current).forward()
        history.append(current)
    return history


def describe_reduced_network():
    """The reduced network's actual weights/biases, for visualizing it --
    mirrors network.describe_network()'s shape, minus the layer3 key."""
    n2_1 = make_neighbor_neuron(threshold=2)
    n2_2 = make_neighbor_neuron(threshold=3)
    n2_3 = make_neighbor_neuron(threshold=4)
    n2_4 = make_center_off_neuron()
    output = make_reduced_output_neuron()

    def spec(neuron):
        return {"weights": neuron.weights, "bias": neuron.bias}

    return {
        "layer2": {"n1": spec(n2_1), "n2": spec(n2_2), "n3": spec(n2_3), "n4": spec(n2_4)},
        "output": spec(output),
    }


if __name__ == "__main__":
    from network import ConwayNet

    mismatches = 0
    for mask in range(512):
        patch = [(mask >> bit) & 1 for bit in range(9)]
        # a 3x3 grid whose center cell has this exact patch as its neighborhood
        grid = [patch[0:3], patch[3:6], patch[6:9]]
        _, _, orig_out = ConwayNet(grid).forward()
        _, reduced_out = ReducedConwayNet(grid).forward()
        if orig_out[1][1] != reduced_out[1][1]:
            mismatches += 1
            print(f"MISMATCH at patch {patch}: original={orig_out[1][1]} reduced={reduced_out[1][1]}")

    print(f"Exhaustive check over all 512 patches: {512 - mismatches}/512 match original ConwayNet.")
