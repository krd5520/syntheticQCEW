import pandas as pd
import numpy as np
from typing import Optional


def find_disclosure_matches(
    basedata: pd.DataFrame,
    listotherdata: list[pd.DataFrame],
    other_names: Optional[list[str]] = None,
    max_similar: int = 3,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    For each dataframe in listotherdata, finds rows where disclosure_code is NA
    in that dataframe but NOT NA in basedata (compare_ids), then finds 1–3
    similar rows (by geography/industry prefix overlap and estab_num proximity)
    where disclosure_code is NA in BOTH datasets.

    Parameters
    ----------
    basedata : pd.DataFrame
        Reference dataframe with columns:
        geography, industry, disclosure_code, estab_num, code.
        (geography, industry) uniquely identifies each row.

    listotherdata : list of pd.DataFrame
        Each dataframe has the same columns as basedata.

    other_names : list of str, optional
        Names for each dataframe in listotherdata (used in the 'source' column).
        Defaults to ["other_0", "other_1", ...].

    max_similar : int
        Maximum number of similar rows to return per compare_id (1–3). Default 3.

    Returns
    -------
    list of (compare_df, similar_df) tuples — one pair per listotherdata entry.

    compare_df columns:
        All original basedata columns + match_id.

    similar_df columns:
        All original columns + match_id + source ("base" or other name) + score.
    """
    if other_names is None:
        other_names = [f"other_{i}" for i in range(len(listotherdata))]

    assert len(other_names) == len(listotherdata), (
        "other_names length must match listotherdata length"
    )

    max_similar = max(1, min(3, max_similar))

    # Index basedata by (geography, industry) for fast lookup
    base_indexed = basedata.set_index(["geography", "industry"])

    results = []

    for other_df, other_name in zip(listotherdata, other_names):
        other_indexed = other_df.set_index(["geography", "industry"])

        # ── Step 1: Find compare_ids ────────────────────────────────────────
        # Rows where disclosure_code IS NA in other but NOT NA in base
        compare_ids = []
        for (geo, ind), other_row in other_indexed.iterrows():
            other_disc = other_row["disclosure_code"]
            is_na_other = pd.isna(other_disc) or str(other_disc).strip().upper() == "NA"
            if not is_na_other:
                continue
            if (geo, ind) not in base_indexed.index:
                continue
            base_disc = base_indexed.loc[(geo, ind), "disclosure_code"]
            is_na_base = pd.isna(base_disc) or str(base_disc).strip().upper() == "NA"
            if not is_na_base:
                compare_ids.append((geo, ind))

        if not compare_ids:
            empty_compare = pd.DataFrame(
                columns=list(basedata.columns) + ["match_id"]
            )
            empty_similar = pd.DataFrame(
                columns=list(basedata.columns) + ["match_id", "source", "score"]
            )
            results.append((empty_compare, empty_similar))
            continue

        # ── Step 2: Pool of candidate similar rows ──────────────────────────
        # Rows where disclosure_code IS NA in BOTH base and other
        candidate_keys = []
        for (geo, ind), base_row in base_indexed.iterrows():
            base_disc = base_row["disclosure_code"]
            is_na_base = pd.isna(base_disc) or str(base_disc).strip().upper() == "NA"
            if not is_na_base:
                continue
            if (geo, ind) not in other_indexed.index:
                continue
            other_disc = other_indexed.loc[(geo, ind), "disclosure_code"]
            is_na_other = pd.isna(other_disc) or str(other_disc).strip().upper() == "NA"
            if is_na_other:
                candidate_keys.append((geo, ind))

        # ── Step 3: Score & match ────────────────────────────────────────────
        compare_rows = []   # rows for compare_df
        similar_rows = []   # rows for similar_df

        for match_id, (cmp_geo, cmp_ind) in enumerate(compare_ids, start=1):
            base_cmp_row = base_indexed.loc[(cmp_geo, cmp_ind)]
            cmp_estab = base_cmp_row["estab_num"]

            scored = []
            for (cand_geo, cand_ind) in candidate_keys:
                score = _similarity_score(
                    cmp_geo, cmp_ind, cmp_estab,
                    cand_geo, cand_ind,
                    base_indexed.loc[(cand_geo, cand_ind), "estab_num"],
                )
                if score > 0:
                    scored.append((score, cand_geo, cand_ind))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_matches = scored[:max_similar]

            # Build compare_df row
            cmp_row_dict = base_indexed.loc[(cmp_geo, cmp_ind)].to_dict()
            cmp_row_dict["geography"] = cmp_geo
            cmp_row_dict["industry"] = cmp_ind
            cmp_row_dict["match_id"] = match_id
            compare_rows.append(cmp_row_dict)

            # Build similar_df rows (base + other for each candidate)
            for score, cand_geo, cand_ind in top_matches:
                # base row
                base_sim = base_indexed.loc[(cand_geo, cand_ind)].to_dict()
                base_sim["geography"] = cand_geo
                base_sim["industry"] = cand_ind
                base_sim["match_id"] = match_id
                base_sim["source"] = "base"
                base_sim["score"] = score
                similar_rows.append(base_sim)

                # other row
                other_sim = other_indexed.loc[(cand_geo, cand_ind)].to_dict()
                other_sim["geography"] = cand_geo
                other_sim["industry"] = cand_ind
                other_sim["match_id"] = match_id
                other_sim["source"] = other_name
                other_sim["score"] = score
                similar_rows.append(other_sim)

        # ── Step 4: Assemble output dataframes ───────────────────────────────
        col_order = list(basedata.columns)

        compare_df = pd.DataFrame(compare_rows)[col_order + ["match_id"]]
        similar_df = (
            pd.DataFrame(similar_rows)[col_order + ["match_id", "source", "score"]]
            if similar_rows
            else pd.DataFrame(columns=col_order + ["match_id", "source", "score"])
        )

        results.append((compare_df, similar_df))

    return results


# ── Scoring helper ────────────────────────────────────────────────────────────

def _prefix_match_length(a: str, b: str) -> int:
    """Return the number of leading characters that match between two strings."""
    a, b = str(a), str(b)
    count = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            count += 1
        else:
            break
    return count


def _similarity_score(
    cmp_geo: str, cmp_ind: str, cmp_estab,
    cand_geo: str, cand_ind: str, cand_estab,
) -> float:
    """
    Compute a similarity score between a compare_id row and a candidate row.

    Score components
    ----------------
    1. Geography prefix overlap  — weighted 2 points per matching leading char
    2. Industry prefix overlap   — weighted 2 points per matching leading char
    3. estab_num proximity       — up to 10 points, decaying as |diff| grows
       score = 10 / (1 + |diff| / max(1, avg_estab))

    Returns 0 if there is no prefix overlap in either geography or industry.
    """
    geo_prefix = _prefix_match_length(cmp_geo, cand_geo)
    ind_prefix = _prefix_match_length(cmp_ind, cand_ind)

    # Require at least one leading character to match in both fields
    if geo_prefix == 0 or ind_prefix == 0:
        return 0.0

    prefix_score = (geo_prefix * 2) + (ind_prefix * 2)

    # estab_num proximity
    try:
        e1, e2 = float(cmp_estab), float(cand_estab)
        avg = max(1.0, (e1 + e2) / 2)
        estab_score = 10.0 / (1.0 + abs(e1 - e2) / avg)
    except (TypeError, ValueError):
        estab_score = 0.0

    return round(prefix_score + estab_score, 4)