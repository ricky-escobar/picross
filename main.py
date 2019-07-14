import os

import numpy as np
from picross import solve
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def parse(filename):
    row_hints = []
    col_hints = []
    cur = row_hints
    with open(filename) as f:
        for line in f.readlines():
            hint = [int(x) for x in line.strip().split()]
            if len(hint) == 0:
                cur = col_hints
            else:
                cur.append([] if hint == [0] else hint)
    return row_hints, col_hints


def show(grid):
    plt.imshow(grid, cmap="Blues", vmin=-1, vmax=1)


def main():
    tests = ["10x10", "15x20"]
    for test in tests:
        row_hints, col_hints = parse(os.path.join("test", test + ".txt"))

        solutions = [grid.copy() for grid in solve(row_hints, col_hints)]

        fig = plt.figure()
        ax = plt.gca()

        x = len(col_hints)
        y = len(row_hints)

        ax.set_xticks(np.arange(-.5, x, 5))
        ax.set_yticks(np.arange(-.5, y, 5))

        ax.set_xticks(np.arange(-.5, x, 1), minor=True)
        ax.set_yticks(np.arange(-.5, y, 1), minor=True)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

        ax.xaxis.tick_top()

        ax.grid(which='minor', color='k', linestyle='-', linewidth=1.5)
        ax.grid(which='major', color='k', linestyle='-', linewidth=3)

        os.makedirs("out", exist_ok=True)
        os.chdir("out")

        show(solutions[-1])
        plt.savefig(f"solution-{test}.png")

        anim = animation.FuncAnimation(fig, show, frames=solutions)
        anim.save(f"solution-{test}.gif", progress_callback=lambda i, n: print(f'{test} gif frame {i}/{n}'))
        anim.save(f"solution-{test}.html", "html", progress_callback=lambda i, n: print(f'{test} html frame {i}/{n}'))

        os.chdir("..")


if __name__ == '__main__':
    main()
