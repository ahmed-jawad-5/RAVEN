"""Shared helper for the attack modules."""
import itertools


def pick_combinations(slots, max_mods, limit, rng):
    """slots: {position: [options]}. Returns up to `limit` lists of (position, option),
    fewest modifications first, shuffled (seeded) within the same size."""
    keys = sorted(slots)
    out = []
    for k in range(1, max_mods + 1):
        group = []
        for chosen in itertools.combinations(keys, k):
            for opts in itertools.product(*(slots[p] for p in chosen)):
                group.append(list(zip(chosen, opts)))
        rng.shuffle(group)
        out.extend(group)
        if len(out) >= limit:
            break
    return out[:limit]
