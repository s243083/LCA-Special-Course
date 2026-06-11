# =============================================================================
# results.py — Output and Reporting
# =============================================================================


import matplotlib.pyplot as plt
import yaml
import os

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    IMPACT_COLUMNS = yaml.safe_load(_f)["IMPACT_COLUMNS"]


def _gwp_label():
    """Return the GWP100 label key from IMPACT_COLUMNS."""
    return list(IMPACT_COLUMNS.keys())[0]


def print_summary_table(aggregated_results, scope):
    """
    Print a GWP100 summary table to the terminal showing emissions
    per life stage and the grand total.

    Args:
        aggregated_results (dict): output of aggregate_by_stage() from lca_engine.py
        scope (str): "per_turbine" or "full_farm"
    """
    by_stage    = aggregated_results["by_stage"]
    grand_total = aggregated_results["grand_total"]
    gwp         = _gwp_label()
    scope_label = "Per Turbine" if scope == "per_turbine" else "Full Farm"

    print("\n" + "=" * 56)
    print(f"  GWP100 RESULTS — {scope_label}")
    print("=" * 56)
    print(f"  {'Life Stage':<28} {'kg CO2-Eq':>22}")
    print("  " + "-" * 52)

    for stage, impacts in by_stage.items():
        val = impacts.get(gwp, 0.0)
        print(f"  {stage:<28} {val:>22.4e}")

    print("  " + "-" * 52)
    print(f"  {'TOTAL':<28} {grand_total.get(gwp, 0.0):>22.4e}")
    print("=" * 56 + "\n")


def print_co2_comparison_table(aggregated_per_turbine, aggregated_full_farm):
    """
    Print a side-by-side CO2 comparison table with life stages as rows
    and per turbine / full farm as columns.

    Args:
        aggregated_per_turbine (dict): output of aggregate_by_stage() for per_turbine scope
        aggregated_full_farm (dict):   output of aggregate_by_stage() for full_farm scope
    """
    gwp        = _gwp_label()
    stages_pt  = aggregated_per_turbine["by_stage"]
    stages_ff  = aggregated_full_farm["by_stage"]
    total_pt   = aggregated_per_turbine["grand_total"].get(gwp, 0.0)
    total_ff   = aggregated_full_farm["grand_total"].get(gwp, 0.0)
    all_stages = list(stages_pt.keys())
    col_w      = 22

    print("\n" + "=" * 72)
    print("  CO2 EMISSIONS SUMMARY (GWP100 — kg CO2-Eq)")
    print("=" * 72)
    print(f"  {'Life Stage':<24} {'Per Turbine':>{col_w}} {'Full Farm':>{col_w}}")
    print("  " + "-" * 68)

    for stage in all_stages:
        val_pt = stages_pt.get(stage, {}).get(gwp, 0.0)
        val_ff = stages_ff.get(stage, {}).get(gwp, 0.0)
        print(f"  {stage:<24} {val_pt:>{col_w}.4e} {val_ff:>{col_w}.4e}")

    print("  " + "-" * 68)
    print(f"  {'TOTAL':<24} {total_pt:>{col_w}.4e} {total_ff:>{col_w}.4e}")
    print("=" * 72 + "\n")


def plot_materials_gwp(aggregated_results, scope):
    """
    Bar chart comparing GWP100 contributions of top-level component groups
    within the Materials life stage only.

    Args:
        aggregated_results (dict): output of aggregate_by_stage() from lca_engine.py
        scope (str): "per_turbine" or "full_farm"
    """
    gwp         = _gwp_label()
    components  = aggregated_results["by_component"]
    scope_label = "Per Turbine" if scope == "per_turbine" else "Full Farm"

    # Filter to Materials stage and aggregate by top-level group
    group_totals = {}
    for comp in components:
        if comp["life_stage"] != "Materials":
            continue
        top_group = comp["path"].split(" > ")[0]
        group_totals[top_group] = group_totals.get(top_group, 0.0) + comp.get(gwp, 0.0)

    if not group_totals:
        print("  [INFO] No Materials data available for chart — skipping.")
        return

    groups = list(group_totals.keys())
    values = list(group_totals.values())
    colors = plt.cm.Set2.colors

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(groups, values, color=colors[:len(groups)], edgecolor="white", linewidth=0.8)

    ax.set_xlabel("Component Group", fontsize=12)
    ax.set_ylabel("GWP100 (kg CO2-Eq)", fontsize=12)
    ax.set_title(f"Materials Stage — GWP100 by Component Group\n({scope_label})",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.2e}",
                ha="center", va="bottom", fontsize=9
            )

    plt.tight_layout()
    plt.show()
    plt.close(fig)



def plot_stage_gwp_pie(aggregated_results, scope):
    """
    Donut chart showing each life stage's percentage contribution to total
    GWP100 emissions. Positive-emission stages form the donut; net-negative
    stages (recycling credits, etc.) are listed as footnotes.

    Args:
        aggregated_results (dict): output of aggregate_by_stage() from lca_engine.py
        scope (str): "per_turbine", "full_farm", or "per_FU"
    """
    gwp         = _gwp_label()
    by_stage    = aggregated_results["by_stage"]
    scope_label = scope.replace("_", " ").title()

    stage_vals  = {stage: impacts.get(gwp, 0.0) for stage, impacts in by_stage.items()}
    positive    = {k: v for k, v in stage_vals.items() if v > 0}
    negative    = {k: v for k, v in stage_vals.items() if v < 0}

    if not positive:
        print("  [INFO] No positive GWP stage data available for pie chart — skipping.")
        return

    labels  = list(positive.keys())
    values  = list(positive.values())
    total   = sum(values)
    palette = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
        "#CCB974", "#64B5CD",
    ]
    colors = palette[:len(labels)]

    fig, ax = plt.subplots(figsize=(9, 7))

    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
        pctdistance=0.75,
    )

    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.text(
        0, 0,
        f"Total\n{total:.2e}\nkg CO2-Eq",
        ha="center", va="center",
        fontsize=10, fontweight="bold", color="#333333",
    )

    legend_labels = [f"{lbl}  ({v:.2e} kg CO2-Eq)" for lbl, v in zip(labels, values)]
    ax.legend(
        wedges, legend_labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.22),
        ncol=2, fontsize=9, frameon=False,
    )

    title = f"GWP100 Contribution by Life Stage\n({scope_label})"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    if negative:
        credit_lines = "  |  ".join(
            f"{s}: {v:.2e}" for s, v in negative.items()
        )
        fig.text(
            0.5, 0.01,
            f"Net credits (not in chart): {credit_lines} kg CO2-Eq",
            ha="center", fontsize=8, color="#666666",
        )

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def print_full_emissions_table(aggregated_results, scope):
    """
    Print a detailed emissions breakdown table with every matched component,
    its quantity, emission factor (GWP100 per unit), total GWP100, subtotals
    per life stage, and a grand total.

    Args:
        aggregated_results (dict): output of aggregate_by_stage() from lca_engine.py
        scope (str): "per_turbine", "full_farm", or "per_FU"
    """
    gwp         = _gwp_label()
    components  = aggregated_results["by_component"]
    by_stage    = aggregated_results["by_stage"]
    grand_total = aggregated_results["grand_total"]
    scope_label = scope.replace("_", " ").title()

    # Group components by life stage, preserving insertion order
    stages_map = {}
    for comp in components:
        stage = comp["life_stage"]
        if stage not in stages_map:
            stages_map[stage] = []
        stages_map[stage].append(comp)

    # Column widths
    W_STAGE = 18
    W_COMP  = 38
    W_QTY   = 14
    W_UNIT  =  6
    W_EF    = 18
    W_TOT   = 18
    SEP     = "  " + "-" * (W_STAGE + W_COMP + W_QTY + W_UNIT + W_EF + W_TOT + 11)

    print("\n" + "=" * (W_STAGE + W_COMP + W_QTY + W_UNIT + W_EF + W_TOT + 13))
    print(f"  FULL EMISSIONS BREAKDOWN — {scope_label}")
    print("=" * (W_STAGE + W_COMP + W_QTY + W_UNIT + W_EF + W_TOT + 13))
    print(
        f"  {'Life Stage':<{W_STAGE}} "
        f"{'Component':<{W_COMP}} "
        f"{'Quantity':>{W_QTY}} "
        f"{'Unit':<{W_UNIT}} "
        f"{'EF (kg CO2/unit)':>{W_EF}} "
        f"{'kg CO2-Eq':>{W_TOT}}"
    )
    print(SEP)

    for stage, comps in stages_map.items():
        stage_label = stage
        stage_subtotal = by_stage.get(stage, {}).get(gwp, 0.0)

        for comp in comps:
            quantity = comp.get("quantity", 0.0) or 0.0
            unit     = comp.get("unit", "")
            gwp_val  = comp.get(gwp, 0.0)
            ef       = (gwp_val / quantity) if quantity else 0.0
            name     = comp["component_name"][:W_COMP]

            print(
                f"  {stage_label:<{W_STAGE}} "
                f"{name:<{W_COMP}} "
                f"{quantity:>{W_QTY}.4e} "
                f"{unit:<{W_UNIT}} "
                f"{ef:>{W_EF}.4e} "
                f"{gwp_val:>{W_TOT}.4e}"
            )
            stage_label = ""  # only print stage name on first row

        # Stage subtotal
        subtotal_label = f"  Subtotal — {stage}"
        print(f"  {'':>{W_STAGE}} {subtotal_label:<{W_COMP + W_QTY + W_UNIT + W_EF + 3}} {stage_subtotal:>{W_TOT}.4e}")
        print(SEP)

    # Grand total
    print(
        f"  {'GRAND TOTAL':<{W_STAGE}} "
        f"{'':>{W_COMP + W_QTY + W_UNIT + W_EF + 3}} "
        f"{grand_total.get(gwp, 0.0):>{W_TOT}.4e}"
    )
    print("=" * (W_STAGE + W_COMP + W_QTY + W_UNIT + W_EF + W_TOT + 13) + "\n")


