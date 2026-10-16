#!/usr/bin/env python3
"""
Created by ChatGPT, checked by thinking meat machines.

Compare private-industry suppression flags in QCEW NAICS quarterly singlefiles.
Python 3.10+, standard library only.

Supported input conventions are checked against each file's actual header.
Identifiers stay strings; populated size codes, including "0", remain in keys.
An absent size_code column or a genuinely blank size field means no size key.
A short/malformed record is an error, NOT an empty/disclosed value.

The uploaded 2015 and 2016 files contain disclosure codes "", "N", and "-".
BLS's quarterly layout documents blank and N but does not define the dash.
Consequently dash handling is an explicit comparison policy:
  exclude         Compare only documented blank <-> N transitions (default).
  not_suppressed  Compare N versus either blank or dash; a user-selected rule,
                  NOT a claim that BLS defines dash as published data.
  error           Refuse comparisons with a matched dash record.
Other codes always raise a contextual error rather than being guessed.

Sources:
https://www.bls.gov/cew/downloadable-data-files.htm
https://www.bls.gov/cew/about-data/downloadable-file-layouts/quarterly/naics-based-quarterly-layout.htm

Validation coverage: real uploaded 2015/2016 archives plus synthetic parser
regression tests. No claim that all historical archives have been tested.
Matching is by literal codes, not a historical NAICS/geography concordance.
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

PRIVATE_OWN_CODE = "5"
SUPPRESSED_CODE = "N"
DashPolicy = Literal["exclude", "not_suppressed", "error"]
REQUIRED_FIELDS = {
    "area_fips", "own_code", "industry_code", "agglvl_code",
    "year", "qtr", "disclosure_code",
}
OUTPUT_FIELDS = [
    "area_fips", "industry_code", "agglvl_code", "size_code", "own_code",
    "old_year", "old_quarter", "old_disclosure_code",
    "new_year", "new_quarter", "new_disclosure_code",
]


@dataclass(frozen=True)
class CellKey:
    area_fips: str
    industry_code: str
    agglvl_code: str
    size_code: str | None = None


@dataclass(frozen=True)
class CellStatus:
    key: CellKey
    disclosure_code: str
    year: int
    quarter: int

    @property
    def suppressed(self) -> bool:
        """Whether the explicit N flag is set; dash policy is applied separately."""
        return self.disclosure_code == SUPPRESSED_CODE


CellMap = dict[CellKey, CellStatus]
Change = tuple[CellStatus, CellStatus]


def validate_year_quarter(year: int, quarter: int) -> None:
    if type(year) is not int or year < 1990:
        raise ValueError(f"Year must be an integer >= 1990; got {year!r}")
    if type(quarter) is not int or quarter not in {1, 2, 3, 4}:
        raise ValueError(f"Quarter must be an integer from 1 to 4; got {quarter!r}")


def validate_state_fips(state_fips: str) -> str:
    if not isinstance(state_fips, str):
        raise ValueError("Use a two-character FIPS string, such as '34'.")
    state_fips = state_fips.strip()
    if len(state_fips) != 2 or any(c not in "0123456789" for c in state_fips):
        raise ValueError(f"Expected a two-digit state FIPS string; got {state_fips!r}")
    return state_fips


def validate_dash_policy(dash_policy: str) -> None:
    if dash_policy not in {"exclude", "not_suppressed", "error"}:
        raise ValueError(f"Unknown dash_policy: {dash_policy!r}")


def qcew_singlefile_url(year: int) -> str:
    validate_year_quarter(year, 1)
    return f"https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip"


def download_if_needed(year: int, cache_dir: Path | str) -> Path:
    """Reuse a cached ZIP; validate new downloads before atomically caching them."""
    url = qcew_singlefile_url(year)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{year}_qtrly_singlefile.zip"
    if destination.exists():
        if not zipfile.is_zipfile(destination):
            raise ValueError(f"Cached file is not a valid ZIP: {destination}")
        return destination
    request = urllib.request.Request(
        url, headers={"User-Agent": "QCEW-suppression-comparator/2.0 (Python)"}
    )
    print(f"Downloading {url}", flush=True)
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".part", delete=False) as output:
            partial = Path(output.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if not zipfile.is_zipfile(partial):
            raise ValueError(f"Response from {url} is not a ZIP archive")
        partial.replace(destination)
        return destination
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def find_singlefile_csv(archive: zipfile.ZipFile, year: int) -> str:
    """Allow partial-year and historical internal names; refuse ambiguous ZIPs."""
    members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
    preferred = [n for n in members if "singlefile" in Path(n).name.lower()]
    year_matches = [n for n in preferred if str(year) in Path(n).name]
    for candidates in (year_matches, preferred, members):
        if len(candidates) == 1:
            return candidates[0]
    raise ValueError(f"Cannot uniquely identify the CSV member: {members!r}")


def normalize_disclosure_code(value: str | None) -> str:
    """Validate a token without erasing the distinction between blank and dash."""
    if value is None:
        raise ValueError("Missing disclosure_code; this is not a blank CSV field")
    if not isinstance(value, str):
        raise ValueError(f"disclosure_code must be text; got {value!r}")
    code = value.strip().upper()
    if code not in {"", "N", "-"}:
        raise ValueError(f"Unexpected disclosure_code {value!r}")
    return code


def text_field(row: Mapping[str, str | None], name: str, *, allow_blank: bool = False) -> str:
    if name not in row or row[name] is None:
        raise ValueError(f"Missing field {name!r}")
    value = row[name]
    if not isinstance(value, str):
        raise ValueError(f"Field {name!r} must be text; got {value!r}")
    value = value.strip()
    if not value and not allow_blank:
        raise ValueError(f"Blank identifier {name!r}")
    return value


def make_cell_key(row: Mapping[str, str | None]) -> CellKey:
    size: str | None = None
    if "size_code" in row:
        size_text = text_field(row, "size_code", allow_blank=True)
        if size_text != "":
            size = size_text  # In particular, keep "0"; do not convert it to blank.
    return CellKey(
        text_field(row, "area_fips"), text_field(row, "industry_code"),
        text_field(row, "agglvl_code"), size,
    )


def load_quarters(
    zip_path: Path | str, year: int, quarters: Iterable[int], state_fips: str,
) -> dict[int, CellMap]:
    """Scan a ZIP once and keep only the requested state/private quarters.

    Use csv.reader for every record, with column indices derived from headers.
    This avoids building millions of dictionaries for irrelevant records, while
    still correctly handling CSV quoting and reordered/additional columns.
    Never stop early based on an assumption that geographic records are sorted.
    """
    zip_path = Path(zip_path)
    state_fips = validate_state_fips(state_fips)
    wanted = set(quarters)
    if not wanted:
        raise ValueError("At least one quarter is required")
    for quarter in wanted:
        validate_year_quarter(year, quarter)
    result: dict[int, CellMap] = {quarter: {} for quarter in wanted}
    with zipfile.ZipFile(zip_path) as archive:
        member = find_singlefile_csv(archive, year)
        source = f"{zip_path.name}:{member}"
        with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            reader = csv.reader(text, strict=True)
            original_header = next(reader, None)
            if original_header is None:
                raise ValueError(f"{source}: empty CSV")
            header = [name.strip().lower() for name in original_header]
            if len(header) != len(set(header)):
                raise ValueError(f"{source}: duplicate column names")
            missing = REQUIRED_FIELDS.difference(header)
            if missing:
                raise ValueError(f"{source}: missing columns {sorted(missing)}")
            index = {name: pos for pos, name in enumerate(header)}
            area_index, own_index = index["area_fips"], index["own_code"]
            width = len(header)
            selected_fields = REQUIRED_FIELDS | ({"size_code"} & index.keys())
            try:
                for values in reader:
                    if not values:  # An empty physical line is not a data record.
                        continue
                    if len(values) != width:
                        raise ValueError(
                            f"{source}, line {reader.line_num}: {len(values)} fields; expected {width}"
                        )
                    if not values[area_index].strip().startswith(state_fips):
                        continue
                    if values[own_index].strip() != PRIVATE_OWN_CODE:
                        continue
                    row = {name: values[index[name]] for name in selected_fields}
                    try:
                        row_year = int(text_field(row, "year"))
                        quarter = int(text_field(row, "qtr"))
                        validate_year_quarter(row_year, quarter)
                        if row_year != year:
                            raise ValueError(f"Expected year {year}, found {row_year}")
                        if quarter not in wanted:
                            continue
                        key = make_cell_key(row)
                        if len(key.area_fips) != 5 or not key.area_fips.isascii() or not key.area_fips.isdigit():
                            raise ValueError(f"Invalid state-prefixed area_fips {key.area_fips!r}")
                        code = normalize_disclosure_code(row["disclosure_code"])
                        if key in result[quarter]:
                            raise ValueError(f"Duplicate key for {year} Q{quarter}: {key}")
                        result[quarter][key] = CellStatus(key, code, year, quarter)
                    except ValueError as error:
                        raise ValueError(
                            f"{source}, line {reader.line_num}, selected fields {row!r}: {error}"
                        ) from error
            except csv.Error as error:
                raise ValueError(f"{source}, line {reader.line_num}: invalid CSV: {error}") from error
    for quarter, cells in sorted(result.items()):
        if not cells:
            raise ValueError(f"{source}: no private/state-{state_fips} rows for {year} Q{quarter}")
        counts = dict(sorted(Counter(c.disclosure_code for c in cells.values()).items()))
        print(f"{year} Q{quarter}: {len(cells):,} rows; disclosure codes {counts!r}", flush=True)
    return result


def load_quarter(zip_path: Path | str, year: int, quarter: int, state_fips: str) -> CellMap:
    """Compatibility wrapper for the original one-quarter interface."""
    return load_quarters(zip_path, year, [quarter], state_fips)[quarter]


def changed_statuses(
    old_cells: CellMap, new_cells: CellMap, *, dash_policy: DashPolicy = "exclude",
) -> tuple[list[Change], list[Change]]:
    """Return (became suppressed, became not suppressed), ignoring absent keys."""
    validate_dash_policy(dash_policy)
    became_suppressed: list[Change] = []
    became_unsuppressed: list[Change] = []
    for key in old_cells.keys() & new_cells.keys():
        old, new = old_cells[key], new_cells[key]
        if "-" in (old.disclosure_code, new.disclosure_code):
            if dash_policy == "error":
                raise ValueError(f"Dash disclosure at matched key {key}; choose an explicit dash policy")
            if dash_policy == "exclude":
                continue
        if not old.suppressed and new.suppressed:
            became_suppressed.append((old, new))
        elif old.suppressed and not new.suppressed:
            became_unsuppressed.append((old, new))

    def order(pair: Change) -> tuple[str, str, str, str]:
        key = pair[0].key
        return (key.area_fips, key.industry_code, key.agglvl_code,
                "" if key.size_code is None else key.size_code)

    became_suppressed.sort(key=order)
    became_unsuppressed.sort(key=order)
    return became_suppressed, became_unsuppressed


def write_changes(changes: Iterable[Change], output_path: Path | str) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for old, new in changes:
            key = old.key
            writer.writerow({
                "area_fips": key.area_fips, "industry_code": key.industry_code,
                "agglvl_code": key.agglvl_code,
                "size_code": "" if key.size_code is None else key.size_code,
                "own_code": PRIVATE_OWN_CODE,
                "old_year": old.year, "old_quarter": old.quarter,
                "old_disclosure_code": old.disclosure_code,
                "new_year": new.year, "new_quarter": new.quarter,
                "new_disclosure_code": new.disclosure_code,
            })
            count += 1
    return count


def compare_loaded_quarters(
    old_cells: CellMap, new_cells: CellMap,
    old_period: tuple[int, int], new_period: tuple[int, int], *,
    output_dir: Path | str, output_prefix: str, dash_policy: DashPolicy = "exclude",
) -> tuple[Path, Path]:
    """Comparison/output step separated from downloading and parsing."""
    became_suppressed, became_unsuppressed = changed_statuses(old_cells, new_cells, dash_policy=dash_policy)
    old_year, old_quarter = old_period
    new_year, new_quarter = new_period
    tag = f"{output_prefix}_{old_year}q{old_quarter}_to_{new_year}q{new_quarter}"
    out = Path(output_dir)
    suppressed = out / f"{tag}_old_not_suppressed_new_suppressed.csv"
    unsuppressed = out / f"{tag}_old_suppressed_new_not_suppressed.csv"
    n_suppressed = write_changes(became_suppressed, suppressed)
    n_unsuppressed = write_changes(became_unsuppressed, unsuppressed)
    common = old_cells.keys() & new_cells.keys()
    dash_pairs = sum("-" in (old_cells[k].disclosure_code, new_cells[k].disclosure_code) for k in common)
    print(
        f"{tag}: {len(common):,} matched; dash_policy={dash_policy!r}; "
        f"{dash_pairs:,} matched pairs involve dash.\n"
        f"  Old only: {len(old_cells.keys() - new_cells.keys()):,}; "
        f"new only: {len(new_cells.keys() - old_cells.keys()):,} (not counted as changes).\n"
        f"  Not suppressed -> suppressed: {n_suppressed:,} -> {suppressed}\n"
        f"  Suppressed -> not suppressed: {n_unsuppressed:,} -> {unsuppressed}", flush=True,
    )
    return suppressed, unsuppressed


def compare_qcew_quarters(
    old_period: tuple[int, int], new_period: tuple[int, int], state_fips: str, *,
    cache_dir: Path | str = "qcew_downloads", output_dir: Path | str = "qcew_results",
    output_prefix: str | None = None, dash_policy: DashPolicy = "exclude",
) -> tuple[Path, Path]:
    """Download/cache inputs, compare the requested periods, and write two CSVs.

    Geography is numeric area_fips beginning with state_fips: state, counties,
    and any state-prefixed residual areas. C-prefixed metro areas are not
    included; assigning overlapping metros requires an explicit area crosswalk.
    """
    validate_year_quarter(*old_period)
    validate_year_quarter(*new_period)
    state_fips = validate_state_fips(state_fips)
    validate_dash_policy(dash_policy)
    requests: dict[int, set[int]] = {}
    for year, quarter in (old_period, new_period):
        requests.setdefault(year, set()).add(quarter)
    data: dict[int, dict[int, CellMap]] = {}
    for year, quarters in sorted(requests.items()):
        path = download_if_needed(year, cache_dir)
        data[year] = load_quarters(path, year, quarters, state_fips)
    return compare_loaded_quarters(
        data[old_period[0]][old_period[1]], data[new_period[0]][new_period[1]],
        old_period, new_period, output_dir=output_dir,
        output_prefix=f"state_{state_fips}" if output_prefix is None else output_prefix,
        dash_policy=dash_policy,
    )


def run_new_jersey_examples(
    cache_dir: Path | str = "qcew_downloads", output_dir: Path | str = "qcew_results", *,
    dash_policy: DashPolicy = "exclude",
) -> None:
    """Hardcoded requested comparisons; scan each annual archive only once."""
    validate_dash_policy(dash_policy)
    data_2015 = load_quarters(download_if_needed(2015, cache_dir), 2015, [4], "34")
    data_2016 = load_quarters(download_if_needed(2016, cache_dir), 2016, [1, 2], "34")
    compare_loaded_quarters(
        data_2015[4], data_2016[1], (2015, 4), (2016, 1),
        output_dir=output_dir, output_prefix="NJ", dash_policy=dash_policy,
    )
    compare_loaded_quarters(
        data_2016[1], data_2016[2], (2016, 1), (2016, 2),
        output_dir=output_dir, output_prefix="NJ", dash_policy=dash_policy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=Path("qcew_downloads"))
    parser.add_argument("--output-dir", type=Path, default=Path("qcew_results"))
    parser.add_argument("--dash-policy", choices=("exclude", "not_suppressed", "error"), default="exclude")
    arguments = parser.parse_args()
    run_new_jersey_examples(arguments.cache_dir, arguments.output_dir, dash_policy=arguments.dash_policy)
