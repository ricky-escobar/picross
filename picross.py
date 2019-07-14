import numpy as np
from functools import reduce


def solve(row_hints, col_hints):
    inference_memo = {}

    def infer(line, hints):
        key = (tuple(hints), tuple(line))
        if key not in inference_memo:
            inference_memo[key] = reduce(np.vectorize(lambda a, b: a if a == b else 0), possibilities(line, hints))
        changed = not np.array_equal(line, inference_memo[key])
        np.copyto(line, inference_memo[key])
        return changed

    grid = np.zeros((len(row_hints), len(col_hints)), np.int8)
    yield grid
    has_changed = True
    while has_changed:
        has_changed = any(infer(grid[i], row_hints[i]) for i in range(len(row_hints))) or \
                      any(infer(grid[:, j], col_hints[j]) for j in range(len(col_hints)))
        yield grid


def possibilities(line, hints, prefix=np.array([], np.int8)):
    if len(hints) == 0:
        prefix_addition = np.full(line.shape, -1, np.int8)
        if not any(line[i] == -prefix_addition[i] for i in range(len(line))):
            yield np.concatenate((prefix, prefix_addition))
    else:
        prefix_addition_size = hints[0] + (len(hints) > 1)
        if len(line) < prefix_addition_size:
            return
        prefix_addition = np.full(prefix_addition_size, -1)
        prefix_addition[:hints[0]].fill(1)
        if not any(line[i] == -prefix_addition[i] for i in range(prefix_addition_size)):
            yield from possibilities(line[prefix_addition_size:], hints[1:], np.concatenate((prefix, prefix_addition)))
        if line[0] != 1:
            yield from possibilities(line[1:], hints, np.concatenate((prefix, [-1])))
