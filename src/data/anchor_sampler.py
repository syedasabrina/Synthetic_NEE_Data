from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

class AnchorSampler:
    """
    Samples real BIPs from the anchor pool for use as topical anchors
    in Module 2 generation. Sampling is stratified by element and score
    level to ensure diversity and prevent anchor reuse within a run.
    """

    def __init__(self, anchor_pool: pd.DataFrame, seed: int = 42,):
        self.pool = anchor_pool.copy()
        self.rng = np.random.default_rng(seed)
        self._used: set[int] = set()

        # index the pool by element and score for fast lookup
        self._index: dict[tuple[str, int], list[int]] = {}
        for (element, score), group in self.pool.groupby(["Element_numberX", "score"]):
            self._index[(element, int(score))] = group.index.tolist()
            

    def sample(self, element: str, score: int, allow_reuse: bool = False) -> pd.Series:
        """
        Returns one anchor BIP for the given element and score level.
        Raises ValueError if no anchors exist for that cell.
        """
        key = (element, score)

        if key not in self._index:
            raise ValueError(
                f"No anchors available for element={element}, score={score}. "
                f"Available cells: {list(self._index.keys())}"
            )

        candidates = self._index[key]

        if not allow_reuse:
            unused = [i for i in candidates if i not in self._used]
            if not unused:
                if score == 0:
                    # score 0 pool is tiny -- allow reuse rather than crash
                    unused = candidates
                else:
                    raise ValueError(
                        f"All anchors exhausted for element={element}, "
                        f"score={score}. Total in cell: {len(candidates)}. "
                        f"Reduce target_per_cell or allow_reuse=True."
                    )

        chosen_idx = self.rng.choice(unused if not allow_reuse else candidates)
        self._used.add(int(chosen_idx))
        return self.pool.loc[chosen_idx]


    def available(self, element: str, score: int) -> int:
        """
        Returns how many unused anchors remain for a given cell.
        Useful for planning how many synthetic BIPs you can generate
        before exhausting the pool.
        """
        key = (element, score)
        if key not in self._index:
            return 0
        candidates = self._index[key]
        unused = [i for i in candidates if i not in self._used]
        return len(unused)


    def reset(self) -> None:
        """
        Clears the used set so anchors can be resampled.
        Call this between generation runs, not within a run.
        """
        self._used.clear()


    def pool_summary(self) -> pd.DataFrame:
        """
        Returns a summary table of anchor counts per element-score cell.
        Use this before a generation run to plan target_per_cell values.
        """
        rows = []
        for (element, score), indices in self._index.items():
            rows.append({
                "element": element,
                "score": score,
                "total_anchors": len(indices),
            })
        return (
            pd.DataFrame(rows)
            .sort_values(["element", "score"])
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    import sys
    from corpus import load, for_anchor_pool

    path = sys.argv[1]
    df = load(path)
    pool = for_anchor_pool(df)

    sampler = AnchorSampler(pool, seed=42)

    print("=== Anchor pool summary ===")
    print(sampler.pool_summary().to_string(index=False))

    print("\n=== Sample one anchor per element at score 4 ===")
    for element in [f"Element{i}" for i in range(1, 8)]:
        row = sampler.sample(element=element, score=4)
        print(f"{element} | tokens: {row['token_count']} | "
              f"preview: {row['Text'][:80]}...")

    print(f"\nUsed so far: {len(sampler._used)} anchors")

    print("\n=== Available anchors remaining (score 4, Element1) ===")
    print(sampler.available("Element1", 4))