"""Optimization log tab for the EOS dashboard.

Displays the per-generation optimization progress log and run-level summary
produced by the GENETIC2 optimiser when ``genetic.log_progress_interval > 0``.

Three sections:

1. **Run Summary** — key metadata cards: device list, forecast statistics,
   convergence generation, total elapsed time, population / horizon / objectives.

2. **Convergence Chart** — dual-line plot (best and mean scalar fitness over
   generations) with a vertical marker at the generation where the best solution
   was last improved.  A secondary axis shows the number of repaired genomes per
   generation as bars.

3. **Per-Objective Chart** — one line per ``obj_*`` column so the contribution
   of each objective (energy cost, LCOS, peak import) can be traced individually.

All charts follow the same Bokeh / MonsterUI styling used in the Plan tab.
"""

from typing import Optional, Union

import requests
from bokeh.models import ColumnDataSource, LinearAxis, Range1d, Span
from bokeh.plotting import figure
from loguru import logger
from monsterui.franken import (
    Card,
    CardTitle,
    Div,
    DivLAligned,
    Grid,
    P,
    Table,
    Tbody,
    Td,
    Th,
    Thead,
    Tr,
    UkIcon,
)

import akkudoktoreos.server.dash.eosstatus as eosstatus
from akkudoktoreos.optimization.optimization import OptimizationSolution
from akkudoktoreos.server.dash.bokeh import Bokeh, bokey_apply_theme_to_plot
from akkudoktoreos.server.dash.components import Error

# ── colour palette (reused from plan.py) ────────────────────────────────────

color_palette = {
    "blue-500":    "#3B82F6",
    "orange-500":  "#F97316",
    "green-500":   "#22C55E",
    "violet-500":  "#8B5CF6",
    "pink-500":    "#EC4899",
    "amber-500":   "#F59E0B",
    "cyan-500":    "#06B6D4",
    "rose-500":    "#F43F5E",
    "lime-500":    "#84CC16",
    "teal-500":    "#14B8A6",
}
_colors = list(color_palette.values())


# ── helpers ──────────────────────────────────────────────────────────────────


def _icon_for_device_type(device_type: str) -> str:
    """Return a MonsterUI icon name that best matches the device type string."""
    t = device_type.lower()
    if "grid" in t:
        return "plug"
    if "inverter" in t or "hybrid" in t:
        return "sun"
    if "appliance" in t:
        return "washing-machine"
    if "fixedload" in t or "load" in t:
        return "zap"
    if "heatpump" in t:
        return "thermometer"
    return "cpu"


def _fmt(value: object, decimals: int = 4) -> str:
    """Format a numeric value for display; fall back to str for non-numerics."""
    try:
        return f"{float(value):.{decimals}f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


# ── Run Summary section ───────────────────────────────────────────────────────


def _RunSummarySection(summary: dict, dark: bool) -> Grid:
    """Build the three-column run-summary card row."""

    # ── Left card: optimiser parameters ─────────────────────────────────────
    params = [
        ("Generations",    summary.get("generations_run",   "—")),
        ("Population",     summary.get("population_size",   "—")),
        ("Horizon steps",  summary.get("horizon",           "—")),
        ("Step interval",  f"{summary.get('step_interval_sec', '—')} s"),
        ("Elapsed",        f"{summary.get('elapsed_sec', '—')} s"),
        ("Converged at",   f"gen {summary.get('best_improved_at_generation', '—')}"),
        ("Best fitness",   _fmt(summary.get("best_scalar_fitness", None))),
        ("Objectives",     ", ".join(summary.get("objective_names", []))),
    ]
    param_rows = [
        Tr(Td(P(k, cls="font-semibold text-sm")), Td(P(str(v), cls="text-sm")))
        for k, v in params
    ]
    params_card = Card(
        Table(Tbody(*param_rows), cls="w-full"),
        header=CardTitle("Optimiser Parameters"),
    )

    # ── Middle card: devices ─────────────────────────────────────────────────
    devices: list[dict] = summary.get("devices", [])
    device_rows = []
    for dev in devices:
        icon = _icon_for_device_type(dev.get("type", ""))
        details = []
        if "inverter_type" in dev:
            details.append(dev["inverter_type"])
        if "battery_capacity_wh" in dev:
            details.append(f"{dev['battery_capacity_wh'] / 1000:.1f} kWh")
        if dev.get("pv_key"):
            details.append(f"PV:{dev['pv_key']}")
        if dev.get("lcos") is not None:
            details.append(f"LCOS:{dev['lcos']}")
        if dev.get("load_key"):
            details.append(f"load:{dev['load_key']}")
        objs = ", ".join(dev.get("objective_names", []))
        device_rows.append(
            Tr(
                Td(DivLAligned(UkIcon(icon=icon), P(dev.get("device_id", "?"), cls="text-sm ml-1"))),
                Td(P(dev.get("type", ""), cls="text-sm text-gray-500")),
                Td(P(", ".join(details), cls="text-sm")),
                Td(P(objs or "—", cls="text-sm text-gray-400")),
            )
        )
    devices_card = Card(
        Table(
            Thead(Tr(Th("Device"), Th("Type"), Th("Params"), Th("Objectives"))),
            Tbody(*device_rows) if device_rows else P("No device info available.", cls="text-sm text-gray-400"),
            cls="w-full text-left",
        ),
        header=CardTitle("Devices"),
    )

    # ── Right card: forecast statistics ─────────────────────────────────────
    forecasts: dict = summary.get("forecasts", {})
    forecast_rows = []
    metric_groups = [
        ("PV power",     "pv_power_w_min",      "pv_power_w_mean",      "pv_power_w_max",      "W"),
        ("Import price", "import_price_min",    "import_price_mean",    "import_price_max",    "amt/kWh"),
    ]
    for label, k_min, k_mean, k_max, unit in metric_groups:
        if k_min in forecasts:
            forecast_rows.append(
                Tr(
                    Td(P(label, cls="font-semibold text-sm")),
                    Td(P(_fmt(forecasts[k_min],  2), cls="text-sm")),
                    Td(P(_fmt(forecasts[k_mean], 2), cls="text-sm font-bold")),
                    Td(P(_fmt(forecasts[k_max],  2), cls="text-sm")),
                    Td(P(unit, cls="text-xs text-gray-400")),
                )
            )
    forecasts_card = Card(
        Table(
            Thead(Tr(Th("Forecast"), Th("Min"), Th("Mean"), Th("Max"), Th("Unit"))),
            Tbody(*forecast_rows) if forecast_rows else P("No forecast info available.", cls="text-sm text-gray-400"),
            cls="w-full text-left",
        ),
        header=CardTitle("Forecast Summary"),
    )

    return Grid(params_card, devices_card, forecasts_card, cols=3)


# ── Convergence chart ─────────────────────────────────────────────────────────


def _ConvergenceChart(df, best_improved_at: Optional[int], dark: bool) -> Div:
    """Dual-axis Bokeh chart: fitness curves + repair counts."""

    # Build source with a plain integer x-axis (generation number).
    # The generation column was reset_index'd to a regular column when
    # we built optimization_log, so it must be present.
    gen_col = "generation" if "generation" in df.columns else df.columns[0]
    gens = df[gen_col].astype(int).tolist()
    source = ColumnDataSource(dict(
        generation=gens,
        best_scalar_fitness=df["best_scalar_fitness"].tolist(),
        mean_scalar_fitness=df["mean_scalar_fitness"].tolist(),
        num_repaired=df["num_repaired"].tolist(),
    ))

    fitness_min = min(
        float(df["best_scalar_fitness"].min()),
        float(df["mean_scalar_fitness"].min()),
    )
    fitness_max = max(
        float(df["best_scalar_fitness"].max()),
        float(df["mean_scalar_fitness"].max()),
    )
    fitness_pad = max((fitness_max - fitness_min) * 0.05, abs(fitness_max) * 0.01, 1e-6)

    repair_max = max(float(df["num_repaired"].max()), 1.0)

    plot = figure(
        title="Convergence: Fitness over Generations",
        x_axis_label="Generation",
        y_axis_label="Scalar Fitness",
        sizing_mode="stretch_width",
        height=320,
        y_range=Range1d(fitness_min - fitness_pad, fitness_max + fitness_pad),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.extra_y_ranges = {"repairs": Range1d(0, repair_max * 1.1)}
    repair_axis = LinearAxis(y_range_name="repairs", axis_label="Repaired Genomes")
    plot.add_layout(repair_axis, "right")

    # Repair bars (behind lines)
    plot.vbar(
        x="generation",
        top="num_repaired",
        width=0.8,
        source=source,
        color="#CBD5E1",  # slate-300 — subtle
        alpha=0.5,
        y_range_name="repairs",
        legend_label="Repaired",
    )

    # Mean fitness (dashed)
    plot.line(
        x="generation",
        y="mean_scalar_fitness",
        source=source,
        line_width=1.5,
        line_dash="dashed",
        color=_colors[0],
        legend_label="Mean fitness",
    )
    # Best fitness (solid, bold)
    plot.line(
        x="generation",
        y="best_scalar_fitness",
        source=source,
        line_width=2.5,
        color=_colors[1],
        legend_label="Best fitness",
    )

    # Vertical marker where best was last improved
    if best_improved_at is not None and best_improved_at in gens:
        conv_span = Span(
            location=best_improved_at,
            dimension="height",
            line_color=_colors[2],
            line_dash="dotted",
            line_width=1.5,
        )
        plot.add_layout(conv_span)
        # Annotate with a text label at the top
        from bokeh.models import Label
        plot.add_layout(Label(
            x=best_improved_at,
            y=fitness_max,
            text=f" converged gen {best_improved_at}",
            text_font_size="11px",
            text_color=_colors[2],
        ))

    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"
    plot.toolbar.autohide = True
    bokey_apply_theme_to_plot(plot, dark)

    return Div(Bokeh(plot))


# ── Per-objective chart ───────────────────────────────────────────────────────


def _ObjectivesChart(df, dark: bool) -> Div:
    """One line per obj_* column so each objective can be traced individually."""

    gen_col = "generation" if "generation" in df.columns else df.columns[0]
    obj_cols = [c for c in df.columns if c.startswith("obj_")]
    if not obj_cols:
        return Div(P("No per-objective data available (all objectives summed into scalar).", cls="text-sm text-gray-400"))

    gens = df[gen_col].astype(int).tolist()
    data: dict = {"generation": gens}
    for col in obj_cols:
        data[col] = df[col].tolist()
    source = ColumnDataSource(data)

    all_vals = [v for col in obj_cols for v in df[col].tolist() if v is not None]
    y_min = min(all_vals) if all_vals else 0.0
    y_max = max(all_vals) if all_vals else 1.0
    y_pad = max((y_max - y_min) * 0.05, abs(y_max) * 0.01, 1e-6)

    plot = figure(
        title="Per-Objective Best Values over Generations",
        x_axis_label="Generation",
        y_axis_label="Objective Value",
        sizing_mode="stretch_width",
        height=280,
        y_range=Range1d(y_min - y_pad, y_max + y_pad),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    for i, col in enumerate(obj_cols):
        label = col[4:]  # strip "obj_"
        plot.line(
            x="generation",
            y=col,
            source=source,
            line_width=2,
            color=_colors[i % len(_colors)],
            legend_label=label,
        )

    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"
    plot.toolbar.autohide = True
    bokey_apply_theme_to_plot(plot, dark)

    return Div(Bokeh(plot))


# ── Top-level tab function ────────────────────────────────────────────────────


def OptLog(
    eos_host: str,
    eos_port: Union[str, int],
    data: Optional[dict] = None,
) -> Div:
    """Generate the Optimization Log tab layout.

    Fetches the optimization solution from the EOS server (reusing the cached
    value when available) and renders three sections:

    1. Run summary cards (parameters, devices, forecasts).
    2. Convergence chart (best/mean fitness + repair counts over generations).
    3. Per-objective chart (individual objective contributions over generations).

    When ``optimization_log`` is ``None`` (i.e. the optimiser ran with
    ``log_progress_interval = 0``) a friendly notice is shown instead of
    the charts, with instructions on how to enable the log.

    Args:
        eos_host: Hostname of the EOS server.
        eos_port: Port of the EOS server.
        data: Optional HTMX POST payload (e.g. dark-mode flag).

    Returns:
        A ``Div`` containing the full tab layout.
    """
    dark = bool(data and data.get("dark") == "true")
    server = f"http://{eos_host}:{eos_port}"

    # ── Fetch solution (reuse cache when fresh) ───────────────────────────
    if eosstatus.eos_solution is None:
        try:
            result = requests.get(
                f"{server}/v1/energy-management/optimization/solution", timeout=10
            )
            result.raise_for_status()
            eosstatus.eos_solution = OptimizationSolution(**result.json())
        except requests.exceptions.HTTPError as e:
            detail = result.json().get("detail", str(e))
            return Error(f"Cannot retrieve optimization solution from {server}: {e}\n{detail}")
        except Exception as e:
            return Error(f"Cannot retrieve optimization solution from {server}: {e}")

    solution: OptimizationSolution = eosstatus.eos_solution

    # ── No log available ──────────────────────────────────────────────────
    if solution.optimization_log is None or solution.run_summary is None:
        return Div(
            Card(
                DivLAligned(
                    UkIcon(icon="info"),
                    P(
                        "No optimization log available. "
                        "Enable it by setting genetic.log_progress_interval > 0 "
                        "in your EOS configuration, e.g.:",
                        cls="ml-2 text-sm",
                    ),
                ),
                P(
                    '{ "optimization": { "genetic": { "log_progress_interval": 10 } } }',
                    cls="font-mono text-xs bg-gray-100 dark:bg-gray-800 p-2 mt-2 rounded",
                ),
            ),
            cls="space-y-4",
        )

    # ── Unpack data ───────────────────────────────────────────────────────
    summary: dict = solution.run_summary
    try:
        log_df = solution.optimization_log.to_dataframe()
    except Exception as e:
        logger.warning(f"OptLog: failed to convert optimization_log to DataFrame: {e}")
        return Error(f"Failed to parse optimization log: {e}")

    if log_df.empty:
        return Error("Optimization log is empty.")

    best_improved_at: Optional[int] = summary.get("best_improved_at_generation")

    # ── Assemble layout ───────────────────────────────────────────────────
    rows = [
        # Run summary
        Card(
            _RunSummarySection(summary, dark),
            header=CardTitle("Run Summary"),
        ),
        # Convergence chart
        Card(
            _ConvergenceChart(log_df, best_improved_at, dark),
            header=CardTitle("Convergence"),
        ),
        # Per-objective chart
        Card(
            _ObjectivesChart(log_df, dark),
            header=CardTitle("Objective Breakdown"),
        ),
    ]

    return Div(*rows, cls="space-y-4")
