import SetCoverPy.setcover as setcover
import numpy as np

def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        covered |= subset

    return cover


def main():
    universe = set(range(1, 11))
    subsets = [set([1, 2, 3, 8, 9, 10]),
               set([1, 2, 3, 4, 5]),
               set([4, 5, 7]),
               set([5, 6, 7]),
               set([6, 7, 8, 9, 10])]

    M = len(universe)
    N = len(subsets)

    a_mat = np.zeros((M, N), dtype=np.bool)
    for s_i, seta in enumerate(subsets):
        for set_v in seta:
            a_mat[set_v-1, s_i] = True
    cost = np.ones(N)/5

    g = setcover.SetCover(a_mat, cost, maxiters=50)
    solution, time_used = g.SolveSCP()
    print(g.s)
    # print(solution)

    # cover = set_cover(universe, subsets)
    # print(cover)


if __name__ == '__main__':
    main()
