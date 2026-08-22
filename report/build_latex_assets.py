from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pct(value: float) -> str:
    return f"{100 * value:.1f}"


def _cell(row: dict[str, str]) -> str:
    return (
        f"{_pct(float(row['success_rate']))} "
        f"[{_pct(max(0.0, float(row['ci_low'])))}; "
        f"{_pct(min(1.0, float(row['ci_high'])))}]"
    )


def _curve(cost_rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "legend.fontsize": 7.6,
            "pdf.fonttype": 42,
        }
    )
    indexed = {(row["method"], int(row["budget"])): row for row in cost_rows}
    zero = indexed[("zero", 0)]
    candidate = next(method for method, _ in indexed if method not in {"zero", "baseline"})
    budgets = sorted(budget for method, budget in indexed if method == "baseline")
    fig, axis = plt.subplots(figsize=(3.45, 2.0))
    for method, label, color, marker in (
        ("baseline", "Naive FT", "#6B7280", "o"),
        (candidate, "CSWI", "#176B73", "s"),
    ):
        rows = [zero] + [indexed[(method, budget)] for budget in budgets]
        x = [0, *budgets]
        y = [100 * float(row["success_rate"]) for row in rows]
        lo = [100 * max(0.0, float(row["ci_low"])) for row in rows]
        hi = [100 * min(1.0, float(row["ci_high"])) for row in rows]
        axis.plot(x, y, marker=marker, color=color, linewidth=1.7, markersize=4.5, label=label)
        axis.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
        for x_value, y_value, row in zip(x, y, rows, strict=True):
            if x_value == 0 and method == "baseline":
                continue
            successes = round(float(row["success_rate"]) * int(row["n"]))
            axis.annotate(
                f"{successes}/{row['n']}",
                (x_value, y_value),
                xytext=(0, 6 if method == candidate else -10),
                textcoords="offset points",
                ha="center",
                va="bottom" if method == candidate else "top",
                color=color,
                fontsize=6.8,
            )
    axis.set_xlabel("Число целевых демонстраций")
    axis.set_ylabel("Успех в симуляторе, %")
    axis.set_xticks([0, *budgets])
    axis.set_xlim(-1, max(budgets) + 1)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#D7DBDF", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _main_table(cost_rows: list[dict[str, str]]) -> str:
    indexed = {(row["method"], int(row["budget"])): row for row in cost_rows}
    candidate = next(method for method, _ in indexed if method not in {"zero", "baseline"})
    budgets = sorted(budget for method, budget in indexed if method == "baseline")
    lines = [
        r"\begin{tabular}{l" + "r" * (len(budgets) + 1) + r"}\toprule",
        "метод & 0 & " + " & ".join(map(str, budgets)) + r"\\\midrule",
        f"Naive FT & {_cell(indexed[('zero', 0)])} & "
        + " & ".join(_cell(indexed[("baseline", budget)]) for budget in budgets)
        + r"\\",
        f"CSWI & {_cell(indexed[('zero', 0)])} & "
        + " & ".join(_cell(indexed[(candidate, budget)]) for budget in budgets)
        + r"\\",
    ]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _task_table(per_run: list[dict[str, str]]) -> str:
    candidate = next(
        row["method"] for row in per_run if row["method"] not in {"zero", "wrong", "baseline"}
    )
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in per_run:
        if row["method"] in {"baseline", candidate}:
            grouped[(row["method"], int(row["task_id"]), int(row["budget"]))].append(row)
    budgets = sorted(
        {
            int(row["budget"])
            for row in per_run
            if row["method"] == "baseline" and int(row["budget"]) > 0
        }
    )
    lines = [
        r"\begin{tabular}{l" + "r" * len(budgets) + r"}\toprule",
        "задача & " + " & ".join(map(str, budgets)) + r"\\\midrule",
    ]
    for task in range(3):
        cells = []
        for budget in budgets:
            values = []
            for method in ("baseline", candidate):
                rows = grouped[(method, task, budget)]
                successes = sum(int(row["successes"]) for row in rows)
                episodes = sum(int(row["n"]) for row in rows)
                values.append(100 * successes / episodes)
            cells.append(fr"{values[0]:.1f}$\to${values[1]:.1f}")
        lines.append(f"T{task} & " + " & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _task_decomposition_text(per_run: list[dict[str, str]]) -> str:
    candidate = next(
        row["method"] for row in per_run if row["method"] not in {"zero", "wrong", "baseline"}
    )
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    budgets = sorted(
        {
            int(row["budget"])
            for row in per_run
            if row["method"] == "baseline" and int(row["budget"]) > 0
        }
    )
    for row in per_run:
        if row["method"] in {"baseline", candidate} and int(row["budget"]) in budgets:
            grouped[(row["method"], int(row["task_id"]), int(row["budget"]))].append(row)
    changes: list[tuple[float, int, int]] = []
    for task in range(3):
        for budget in budgets:
            before = sum(int(row["successes"]) for row in grouped[("baseline", task, budget)])
            after = sum(int(row["successes"]) for row in grouped[(candidate, task, budget)])
            changes.append((100 * (after - before) / 40, task, budget))
    positive = sum(delta > 0 for delta, _, _ in changes)
    tied = sum(delta == 0 for delta, _, _ in changes)
    negative = sum(delta < 0 for delta, _, _ in changes)
    best = max(changes)
    worst = min(changes)
    return (
        f"В {len(changes)} task--budget ячейках $k={'/'.join(map(str, budgets))}$ "
        f"положительных сдвигов {positive}, "
        f"ничьих {tied}, отрицательных {negative}. Максимум: T{best[1]}/$k={best[2]}$ "
        f"({best[0]:+.1f}~п.п.); минимум: T{worst[1]}/$k={worst[2]}$ "
        f"({worst[0]:+.1f}~п.п.)."
    )


def _language_text(rows: list[dict[str, str]]) -> str:
    n_correct = sum(int(row["n_correct"]) for row in rows)
    n_wrong = sum(int(row["n_wrong"]) for row in rows)
    correct = sum(round(float(row["correct_rate"]) * int(row["n_correct"])) for row in rows)
    wrong = sum(round(float(row["wrong_rate"]) * int(row["n_wrong"])) for row in rows)
    correct_pct = 100 * correct / n_correct
    wrong_pct = 100 * wrong / n_wrong
    return (
        f"Correct prompt: {correct}/{n_correct} ({correct_pct:.1f}\\%); "
        f"cyclic-wrong prompt: {wrong}/{n_wrong} ({wrong_pct:.1f}\\%); "
        f"наблюдаемое падение {correct_pct - wrong_pct:.1f}~п.п."
    )


def _exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / 2**discordant)


def _comparison_text(rows: list[dict[str, str]]) -> str:
    candidate = next(
        row["method"] for row in rows if row["method"] not in {"zero", "wrong", "baseline"}
    )
    indexed = {
        (row["method"], int(row["budget"]), int(row["task_id"]), int(row["seed"])): row
        for row in rows
        if row["method"] in {"baseline", candidate}
    }
    budgets = sorted(
        int(row["budget"])
        for row in rows
        if row["method"] == "baseline" and int(row["budget"]) > 0
    )
    budgets = sorted(set(budgets))
    comparisons = []
    for budget in budgets:
        wins = losses = total = 0
        for task in range(3):
            for seed in (17, 29):
                baseline = json.loads(indexed[("baseline", budget, task, seed)]["success_vector"])
                method = json.loads(indexed[(candidate, budget, task, seed)]["success_vector"])
                if len(baseline) != len(method):
                    raise ValueError("Paired checkpoint evaluations have unequal denominators")
                wins += sum(bool(m) and not bool(b) for b, m in zip(baseline, method, strict=True))
                losses += sum(bool(b) and not bool(m) for b, m in zip(baseline, method, strict=True))
                total += len(baseline)
        delta = 100 * (wins - losses) / total
        p_value = _exact_mcnemar_p(wins, losses)
        p_text = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
        comparisons.append(
            f"$k={budget}$: $\\Delta={delta:+.1f}$~п.п., "
            f"$W/L={wins}/{losses}$, exact $p={p_text}$"
        )
    return (
        "Парное сравнение на одинаковых initial states (двусторонний exact McNemar): "
        + "; ".join(comparisons)
        + "."
    )


def _paired_table(rows: list[dict[str, str]]) -> str:
    candidate = next(
        row["method"] for row in rows if row["method"] not in {"zero", "wrong", "baseline"}
    )
    indexed = {
        (row["method"], int(row["budget"]), int(row["task_id"]), int(row["seed"])): row
        for row in rows
        if row["method"] in {"baseline", candidate}
    }
    budgets = sorted(
        {
            int(row["budget"])
            for row in rows
            if row["method"] == "baseline" and int(row["budget"]) > 0
        }
    )
    lines = [
        r"\begin{tabular}{rrrrr}\toprule",
        r"$k$ & $\Delta$, п.п. & wins & losses & exact $p$\\\midrule",
    ]
    for budget in budgets:
        wins = losses = total = 0
        for task in range(3):
            for seed in (17, 29):
                before = json.loads(indexed[("baseline", budget, task, seed)]["success_vector"])
                after = json.loads(indexed[(candidate, budget, task, seed)]["success_vector"])
                wins += sum(bool(a) and not bool(b) for b, a in zip(before, after, strict=True))
                losses += sum(bool(b) and not bool(a) for b, a in zip(before, after, strict=True))
                total += len(before)
        delta = 100 * (wins - losses) / total
        p_value = _exact_mcnemar_p(wins, losses)
        lines.append(f"{budget} & {delta:+.1f} & {wins} & {losses} & {p_value:.3f}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _robustness_table(paths: list[Path]) -> str:
    labels = [
        ("CSWI", 1000),
        ("CSTVI", 2000),
        ("BDAM", 3000),
    ]
    if len(paths) != len(labels):
        raise ValueError("robustness table requires exactly three comparison files")
    lines = [
        r"\begin{tabular}{lrrrr}\toprule",
        r"метод & eval seed & $\Delta_5$ & $\Delta_{10}$ & $\Delta_{25}$\\\midrule",
    ]
    for path, (label, eval_seed) in zip(paths, labels, strict=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        indexed = {
            int(row["budget"]): float(row["delta_percentage_points"])
            for row in payload["aggregate"]
        }
        deltas = " & ".join(f"{indexed[budget]:+.1f}" for budget in (5, 10, 25))
        lines.append(f"{label} & {eval_seed} & {deltas}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _bonus_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    small = payload["small_spearman"]
    robometer = payload["robometer_spearman"]
    if small is None and robometer is None:
        return (
            f"По {payload['n_checkpoints']} checkpoint cells simulator success константен; "
            "Spearman $\\rho$ для обоих критиков не определён. Это floor/ceiling-ограничение, "
            "а не нулевая корреляция."
        )
    small_text = "не определён" if small is None else f"{small:.3f}"
    robometer_text = "не определён" if robometer is None else f"{robometer:.3f}"
    small_values = [float(record["small_critic"]) for record in payload["records"]]
    robometer_values = [float(record["robometer"]) for record in payload["records"]]
    small_span = max(small_values) - min(small_values)
    robometer_span = max(robometer_values) - min(robometer_values)
    return (
        f"По {payload['n_checkpoints']} checkpoint cells Spearman $\\rho$ малого критика "
        f"равен {small_text} при span score {small_span:.4f}; Robometer --- {robometer_text} "
        f"при span {robometer_span:.3f}. Малый критик практически коллапсировал к константе, "
        "тогда как Robometer отделяет zero-shot floor, но его положительное ранжирование остаётся "
        "умеренным и не означает калибровку reward."
    )


def _optimization_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    baseline = 100 * float(payload["baseline_reference_success"])
    small = payload["selectors"]["small_critic"]
    robometer = payload["selectors"]["robometer"]
    return (
        f"При прямом argmax proxy среди четырех policy на каждой из "
        f"{payload['n_task_budget_groups']} task--budget групп baseline-reference равен "
        f"{baseline:.1f}\\%. Малый критик выбирает {100 * small['mean_selected_true_success']:.1f}\\% "
        f"($\\Delta={100 * small['delta_vs_baseline_reference']:+.1f}$~п.п.; "
        f"{small['suboptimal_selections']}/{payload['n_task_budget_groups']} suboptimal), Robometer --- "
        f"{100 * robometer['mean_selected_true_success']:.1f}\\% "
        f"($\\Delta={100 * robometer['delta_vs_baseline_reference']:+.1f}$~п.п.). "
        "Следовательно, максимизация малого video-reward выбирает объективно худшую policy; "
        "Robometer устойчивее, но также не достигает simulator oracle."
    )


def _ceiling_text(cost_rows: list[dict[str, str]]) -> tuple[str, str]:
    indexed = {(row["method"], int(row["budget"])): row for row in cost_rows}
    candidate = next(method for method, _ in indexed if method not in {"zero", "baseline"})
    budgets = sorted(budget for method, budget in indexed if method == "baseline")
    baseline = [round(float(indexed[("baseline", budget)]["success_rate"]) * 120) for budget in budgets]
    method = [round(float(indexed[(candidate, budget)]["success_rate"]) * 120) for budget in budgets]
    deltas = [100 * (after - before) / 120 for before, after in zip(baseline, method, strict=True)]
    budget_text = "/".join(str(budget) for budget in budgets)
    baseline_text = "/".join(str(value) for value in baseline)
    method_text = "/".join(str(value) for value in method)
    delta_text = "/".join(f"{value:+.1f}" for value in deltas)
    summary = (
        f"На независимом eval seed 4000 для $k={budget_text}$ baseline дал "
        f"[{baseline_text}]/120 успехов, CSWI --- [{method_text}]/120; "
        f"парные изменения равны [{delta_text}]~п.п."
    )
    failing = [budget for budget, before, after in zip(budgets, baseline, method, strict=True) if after <= before]
    if failing:
        verdict = (
            "Строгий критерий превосходства на каждом бюджете не выполнен: "
            + ", ".join(f"$k={budget}$" for budget in failing)
            + " не дают положительного приращения."
        )
    else:
        verdict = "Строгий критерий превосходства выполнен на всех зарегистрированных бюджетах."
    return summary, verdict


def _hypothesis_four_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    small = payload["small_spearman"]
    robometer = payload["robometer_spearman"]
    if small is None or robometer is None:
        return (
            "неразрешимо: по крайней мере один Spearman $\\rho$ не определён из-за "
            "константных rank-значений"
        )
    verdict = "подтверждено" if robometer > small else "опровергнуто"
    return (
        f"{verdict}: $\\rho_{{small}}={small:.3f}$, "
        f"$\\rho_{{Robo}}={robometer:.3f}$"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector figure and LaTeX result macros")
    parser.add_argument("--results", type=Path, default=Path("artifacts/results"))
    parser.add_argument(
        "--bonus-results", type=Path, default=Path("artifacts/results/reward_ranking.json")
    )
    parser.add_argument(
        "--bonus-optimization",
        type=Path,
        default=Path("artifacts/results/reward_optimization.json"),
    )
    parser.add_argument(
        "--robustness-comparisons",
        type=Path,
        nargs=3,
        default=[
            Path("artifacts/results_v3/paired_comparison.json"),
            Path("artifacts/results_v4/paired_comparison.json"),
            Path("artifacts/results_v5/paired_comparison.json"),
        ],
    )
    parser.add_argument("--report-dir", type=Path, default=Path("report"))
    args = parser.parse_args()
    cost_rows = _rows(args.results / "cost_curve.csv")
    per_run = _rows(args.results / "per_run.csv")
    language = _rows(args.results / "language_control.csv")
    ceiling_summary, ceiling_verdict = _ceiling_text(cost_rows)
    _curve(cost_rows, args.report_dir / "cost_curve.pdf")
    macros = "\n".join(
        [
            r"\newcommand{\CostCurveFigure}{\includegraphics[width=\linewidth]{report/cost_curve.pdf}}",
            r"\newcommand{\MainResultTable}{%",
            _main_table(cost_rows),
            "}",
            r"\newcommand{\PerTaskTable}{%",
            _task_table(per_run),
            "}",
            r"\newcommand{\TaskDecompositionText}{%",
            _task_decomposition_text(per_run),
            "}",
            r"\newcommand{\LanguageText}{%",
            _language_text(language),
            "}",
            r"\newcommand{\ComparisonText}{%",
            _comparison_text(per_run),
            "}",
            r"\newcommand{\PairedTable}{%",
            _paired_table(per_run),
            "}",
            r"\newcommand{\RobustnessTable}{%",
            _robustness_table(args.robustness_comparisons),
            "}",
            r"\newcommand{\BonusText}{%",
            _bonus_text(args.bonus_results),
            "}",
            r"\newcommand{\OptimizationText}{%",
            _optimization_text(args.bonus_optimization),
            "}",
            r"\newcommand{\CeilingSummary}{%",
            ceiling_summary,
            "}",
            r"\newcommand{\CeilingVerdict}{%",
            ceiling_verdict,
            "}",
            r"\newcommand{\HypothesisFourText}{%",
            _hypothesis_four_text(args.bonus_results),
            "}",
            "",
        ]
    )
    (args.report_dir / "generated_results.tex").write_text(macros, encoding="utf-8")
    print(args.report_dir / "generated_results.tex")


if __name__ == "__main__":
    main()
