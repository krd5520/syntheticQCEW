import os
import numpy as np
from typing import Optional


def regression_table(
    models: list,
    model_names: Optional[list[str]] = None,
    dep_var_label: str = "Dependent variable:",
    sig_levels: tuple = (0.01, 0.05, 0.10),
    sig_stars: tuple = ("***", "**", "*"),
    float_fmt: str = "{:.3f}",
    title: Optional[str] = None,
    covariate_labels: Optional[dict] = None,
    latex_path: Optional[str] = None,
) -> str:
    """
    Produces a Stargazer-style regression summary table from a list of
    statsmodels fitted models.  Optionally saves a LaTeX version to disk.

    Parameters
    ----------
    models : list
        List of fitted statsmodels regression results (e.g. OLS, WLS, GLS).
    model_names : list[str], optional
        Column headers for each model. Defaults to "(1)", "(2)", ...
    dep_var_label : str
        Label printed above the dependent variable names row.
    sig_levels : tuple
        Significance thresholds in ascending order.
    sig_stars : tuple
        Star strings corresponding to each threshold.
    float_fmt : str
        Python format string for numeric values.
    title : str, optional
        Optional table title printed at the top (also used as LaTeX caption).
    covariate_labels : dict, optional
        Mapping from original parameter name to a display label,
        e.g. {"x1": "Income", "Intercept": "Constant"}.
    latex_path : str, optional
        File path where the LaTeX table should be saved, e.g.
        "results/table1.tex".  If the directory does not exist it is created.
        The file contains a self-contained ``table`` + ``tabular`` environment
        that can be \\input{} directly into any LaTeX document.

    Returns
    -------
    str
        A plain-text table ready for printing.
    """
    if not models:
        raise ValueError("models list is empty.")

    n_models = len(models)

    # ------------------------------------------------------------------ #
    # 1. Build column headers                                              #
    # ------------------------------------------------------------------ #
    if model_names is None:
        model_names = [f"({i + 1})" for i in range(n_models)]
    if len(model_names) != n_models:
        raise ValueError("Length of model_names must match number of models.")

    # ------------------------------------------------------------------ #
    # 2. Collect all predictor names; ensure Intercept is first           #
    # ------------------------------------------------------------------ #
    all_params: list[str] = []
    for m in models:
        for p in m.params.index:
            if p not in all_params:
                all_params.append(p)

    intercept_key = next(
        (p for p in all_params if p.lower() in ("intercept", "const")), None
    )
    if intercept_key:
        all_params.remove(intercept_key)
        all_params.insert(0, intercept_key)

    # ------------------------------------------------------------------ #
    # 3. Helper: significance stars                                        #
    # ------------------------------------------------------------------ #
    def get_stars(pval: float) -> str:
        for thresh, star in zip(sig_levels, sig_stars):
            if pval < thresh:
                return star
        return ""

    # ------------------------------------------------------------------ #
    # 4. Helper: display name for a parameter                             #
    # ------------------------------------------------------------------ #
    def display_name(p: str) -> str:
        if covariate_labels and p in covariate_labels:
            return covariate_labels[p]
        return p

    # ------------------------------------------------------------------ #
    # 5. Build cell contents for each (param, model) pair                 #
    # ------------------------------------------------------------------ #
    cells: dict[str, list[Optional[tuple[str, str]]]] = {}
    for p in all_params:
        cells[p] = []
        for m in models:
            if p in m.params.index:
                coef = m.params[p]
                se = m.bse[p]
                pval = m.pvalues[p]
                coef_str = float_fmt.format(coef) + get_stars(pval)
                se_str = f"({float_fmt.format(se)})"
                cells[p].append((coef_str, se_str))
            else:
                cells[p].append(None)

    # ------------------------------------------------------------------ #
    # 6. Collect footer stats                                              #
    # ------------------------------------------------------------------ #
    obs_row: list[str] = []
    fstat_row: list[str] = []
    r2_adj_row: list[str] = []

    for m in models:
        obs_row.append(str(int(m.nobs)))

        try:
            fval = m.fvalue
            df_model = int(m.df_model)
            df_resid = int(m.df_resid)
            fstat_row.append(
                f"{float_fmt.format(fval)} (df={df_model}; {df_resid})"
            )
        except AttributeError:
            fstat_row.append("N/A")

        try:
            r2_adj_row.append(float_fmt.format(m.rsquared_adj))
        except AttributeError:
            r2_adj_row.append("N/A")

    # ------------------------------------------------------------------ #
    # 7. Dependent variable names                                          #
    # ------------------------------------------------------------------ #
    dep_vars: list[str] = []
    for m in models:
        try:
            dep_vars.append(str(m.model.endog_names))
        except AttributeError:
            dep_vars.append("")

    # ------------------------------------------------------------------ #
    # 8. Compute column widths (plain-text table)                         #
    # ------------------------------------------------------------------ #
    label_width = max(
        max(len(display_name(p)) for p in all_params),
        len("Observations"),
        len("F Statistic"),
        len("Adjusted R\u00b2"),
        14,
    )

    col_widths: list[int] = []
    for idx, m_name in enumerate(model_names):
        w = len(m_name)
        w = max(w, len(dep_vars[idx]))
        for p in all_params:
            cell = cells[p][idx]
            if cell:
                w = max(w, len(cell[0]), len(cell[1]))
        w = max(w, len(obs_row[idx]), len(fstat_row[idx]), len(r2_adj_row[idx]))
        col_widths.append(max(w, 10))

    total_width = label_width + sum(col_widths) + 3 * n_models + 1

    # ------------------------------------------------------------------ #
    # 9. Plain-text rendering helpers                                      #
    # ------------------------------------------------------------------ #
    HLINE = "=" * total_width
    THIN  = "-" * total_width

    def row(label: str, values: list[str], center: bool = True) -> str:
        label_cell = label.ljust(label_width)
        cols = ""
        for val, w in zip(values, col_widths):
            cols += " " + (val.center(w) if center else val.ljust(w)) + " |"
        return f"|{label_cell}|{cols}"

    def empty_row() -> str:
        return row("", [""] * n_models)

    # ------------------------------------------------------------------ #
    # 10. Assemble plain-text table                                        #
    # ------------------------------------------------------------------ #
    lines: list[str] = []

    if title:
        lines.append(title.center(total_width))

    lines.append(HLINE)
    lines.append(row("", model_names))
    lines.append(row(dep_var_label, dep_vars))
    lines.append(THIN)

    for p in all_params:
        coef_vals = [c[0] if c else "" for c in cells[p]]
        se_vals   = [c[1] if c else "" for c in cells[p]]
        lines.append(row(display_name(p), coef_vals))
        lines.append(row("", se_vals))
        lines.append(empty_row())

    lines.append(THIN)
    lines.append(row("Observations", obs_row))
    lines.append(row("F Statistic",  fstat_row))
    lines.append(row("Adjusted R\u00b2", r2_adj_row))
    lines.append(HLINE)

    note_parts = [f"p<{t}: {s}" for t, s in zip(sig_levels, sig_stars)]
    lines.append("Note: " + ";  ".join(note_parts))

    plain_text = "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 11. Build and save LaTeX table (if latex_path provided)             #
    # ------------------------------------------------------------------ #
    if latex_path is not None:
        latex_str = _build_latex(
            all_params=all_params,
            model_names=model_names,
            dep_vars=dep_vars,
            dep_var_label=dep_var_label,
            cells=cells,
            obs_row=obs_row,
            fstat_row=fstat_row,
            r2_adj_row=r2_adj_row,
            n_models=n_models,
            display_name=display_name,
            sig_levels=sig_levels,
            sig_stars=sig_stars,
            title=title,
        )
        dest = os.path.expanduser(latex_path)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(latex_str)
        print(f"LaTeX table saved to: {dest}")

    return plain_text


# ======================================================================== #
# LaTeX builder                                                             #
# ======================================================================== #
def _build_latex(
    *,
    all_params,
    model_names,
    dep_vars,
    dep_var_label,
    cells,
    obs_row,
    fstat_row,
    r2_adj_row,
    n_models,
    display_name,
    sig_levels,
    sig_stars,
    title,
) -> str:
    """Return a complete LaTeX table string."""

    def tex_escape(s: str) -> str:
        """Escape characters that are special in LaTeX."""
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
            "\\": r"\textbackslash{}",
        }
        # Stars (*) are fine in LaTeX math/text mode — leave them alone.
        for char, repl in replacements.items():
            s = s.replace(char, repl)
        return s

    def cell_to_tex(val: str) -> str:
        """Wrap a coefficient string so stars use $^{}$ superscripts."""
        if not val:
            return ""
        # Separate trailing stars from the number
        stripped = val.rstrip("*")
        star_part = val[len(stripped):]
        if star_part:
            return f"{tex_escape(stripped)}$^{{{star_part}}}$"
        return tex_escape(stripped)

    col_spec = "l" + "c" * n_models           # l for labels, c for each model
    col_sep  = " & "
    nl       = " \\\\\n"

    L: list[str] = []

    # Preamble comment
    L.append("% Regression table generated by regression_table()")
    L.append("% Requires: booktabs (for \\toprule etc.) — add to preamble:")
    L.append("%   \\usepackage{booktabs}")
    L.append("")

    L.append("\\begin{table}[htbp]")
    L.append("  \\centering")

    if title:
        L.append(f"  \\caption{{{tex_escape(title)}}}")

    L.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    L.append("    \\toprule")

    # ---- model name header ----
    header_cells = [""] + [tex_escape(n) for n in model_names]
    L.append("    " + col_sep.join(header_cells) + nl.rstrip("\n"))

    # ---- dependent variable row ----
    dv_cells = [tex_escape(dep_var_label)] + [tex_escape(d) for d in dep_vars]
    L.append("    " + col_sep.join(dv_cells) + nl.rstrip("\n"))
    L.append("    \\midrule")

    # ---- predictor rows ----
    for p in all_params:
        label = tex_escape(display_name(p))

        coef_cells = [label]
        se_cells   = [""]

        for cell in cells[p]:
            if cell:
                coef_cells.append(cell_to_tex(cell[0]))
                se_cells.append(tex_escape(cell[1]))
            else:
                coef_cells.append("")
                se_cells.append("")

        L.append("    " + col_sep.join(coef_cells) + nl.rstrip("\n"))
        L.append("    " + col_sep.join(se_cells)   + nl.rstrip("\n"))
        L.append("    " + col_sep.join([""] * (n_models + 1)) + nl.rstrip("\n"))

    # ---- footer ----
    L.append("    \\midrule")

    obs_cells  = ["Observations"] + [tex_escape(v) for v in obs_row]
    fstat_cells = ["F Statistic"]  + [tex_escape(v) for v in fstat_row]
    r2_cells   = ["Adjusted $R^{2}$"] + [tex_escape(v) for v in r2_adj_row]

    L.append("    " + col_sep.join(obs_cells)  + nl.rstrip("\n"))
    L.append("    " + col_sep.join(fstat_cells) + nl.rstrip("\n"))
    L.append("    " + col_sep.join(r2_cells)   + nl.rstrip("\n"))
    L.append("    \\bottomrule")

    # ---- significance note as multicolumn footnote ----
    note_parts = [f"p$<${t}: {s}" for t, s in zip(sig_levels, sig_stars)]
    note_str   = "\\textit{Note:} " + ";\\; ".join(note_parts)
    L.append(
        f"    \\multicolumn{{{n_models + 1}}}{{r}}{{{note_str}}} \\\\"
    )

    L.append("  \\end{tabular}")
    L.append("\\end{table}")

    return "\n".join(L) + "\n"


# ======================================================================== #
# Demo                                                                      #
# ======================================================================== #
# if __name__ == "__main__":
#     import statsmodels.api as sm
#     import pandas as pd
#
#     rng = np.random.default_rng(42)
#     n = 200
#
#     x1 = rng.normal(size=n)
#     x2 = rng.normal(size=n)
#     x3 = rng.normal(size=n)
#     y  = 2.5 + 1.2 * x1 - 0.8 * x2 + 0.4 * x3 + rng.normal(scale=1.5, size=n)
#
#     df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
#
#     m1 = sm.OLS(df["y"], sm.add_constant(df[["x1"]])).fit()
#     m2 = sm.OLS(df["y"], sm.add_constant(df[["x1", "x2"]])).fit()
#     m3 = sm.OLS(df["y"], sm.add_constant(df[["x1", "x2", "x3"]])).fit()
#
#     table = regression_table(
#         models=[m1, m2, m3],
#         model_names=["Model 1", "Model 2", "Model 3"],
#         title="OLS Regression Results",
#         covariate_labels={"const": "Constant"},
#         latex_path="output/table1.tex",   # <-- set to None to skip saving
#     )
#     print(table)
