from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/vla_budget_report.pdf"
RESULTS = ROOT / "artifacts/results"
TMP = ROOT / "tmp/pdfs/vla_budget_report"
BLACK = colors.black
DARK = colors.HexColor("#303030")
MID = colors.HexColor("#6A6A6A")
LIGHT = colors.HexColor("#D0D0D0")
PALE = colors.HexColor("#F2F2F2")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required measured result is missing: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_predictions(path: Path) -> dict:
    """Read the small fixed preregistration schema without a runtime YAML dependency."""
    text = path.read_text(encoding="utf-8")
    zero = int(re.search(r"zero_shot_percent:\s*(\d+)", text).group(1))
    output = {"zero_shot_percent": zero}
    for name in ("baseline_percent", "method_percent"):
        match = re.search(rf"{name}:\s*\{{5:\s*(\d+),\s*10:\s*(\d+),\s*25:\s*(\d+)\}}", text)
        output[name] = dict(zip((5, 10, 25), map(int, match.groups()), strict=True))
    return {"expected_shape": output}


def register_fonts() -> None:
    font_sets = [
        {
            "Serif": Path("C:/Windows/Fonts/times.ttf"),
            "Serif-Bold": Path("C:/Windows/Fonts/timesbd.ttf"),
            "Sans": Path("C:/Windows/Fonts/arial.ttf"),
            "Sans-Bold": Path("C:/Windows/Fonts/arialbd.ttf"),
        },
        {
            "Serif": Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
            "Serif-Bold": Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
            "Sans": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            "Sans-Bold": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        },
    ]
    for font_set in font_sets:
        if all(path.exists() for path in font_set.values()):
            for name, path in font_set.items():
                pdfmetrics.registerFont(TTFont(name, path))
            return
    raise FileNotFoundError("Times New Roman/Arial or DejaVu fonts are required")


def make_styles():
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=sample["Title"], fontName="Sans-Bold", fontSize=19,
            leading=22, alignment=TA_LEFT, textColor=BLACK, spaceAfter=3 * mm,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=sample["Normal"], fontName="Sans", fontSize=9,
            leading=12, textColor=DARK, spaceAfter=4 * mm,
        ),
        "h1": ParagraphStyle(
            "h1", parent=sample["Heading1"], fontName="Sans-Bold", fontSize=12,
            leading=14, textColor=BLACK, spaceBefore=2 * mm, spaceAfter=1.5 * mm,
        ),
        "h2": ParagraphStyle(
            "h2", parent=sample["Heading2"], fontName="Sans-Bold", fontSize=9.2,
            leading=11, textColor=BLACK, spaceBefore=1.4 * mm, spaceAfter=0.8 * mm,
        ),
        "body": ParagraphStyle(
            "body", parent=sample["BodyText"], fontName="Serif", fontSize=8.4,
            leading=11.2, textColor=BLACK, spaceAfter=1.5 * mm,
        ),
        "small": ParagraphStyle(
            "small", parent=sample["BodyText"], fontName="Serif", fontSize=6.8,
            leading=8.5, textColor=DARK,
        ),
        "caption": ParagraphStyle(
            "caption", parent=sample["BodyText"], fontName="Serif", fontSize=6.5,
            leading=8, textColor=DARK, alignment=TA_CENTER,
        ),
        "abstract": ParagraphStyle(
            "abstract", parent=sample["BodyText"], fontName="Serif", fontSize=8.2,
            leading=10.8, leftIndent=8 * mm, rightIndent=8 * mm, textColor=BLACK,
            spaceAfter=2 * mm,
        ),
    }


def page_frame(canvas, document):
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(BLACK)
    canvas.setLineWidth(0.45)
    canvas.line(15 * mm, height - 11 * mm, width - 15 * mm, height - 11 * mm)
    canvas.setFont("Sans", 6.5)
    canvas.setFillColor(DARK)
    canvas.drawString(15 * mm, height - 8 * mm, "SmolVLA low-demo adaptation on LIBERO")
    canvas.line(15 * mm, 10 * mm, width - 15 * mm, 10 * mm)
    canvas.drawString(15 * mm, 6.5 * mm, "Measured simulator results; compute-limited protocol disclosed")
    canvas.drawRightString(width - 15 * mm, 6.5 * mm, str(document.page))
    canvas.restoreState()


def table(data, widths, *, font_size=6.9, header=True):
    result = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Serif"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E4E4E4")),
            ("FONTNAME", (0, 0), (-1, 0), "Sans-Bold"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, BLACK),
        ]
    result.setStyle(TableStyle(commands))
    return result


class CostCurve(Flowable):
    def __init__(self, measured: list[dict[str, str]], predicted: dict, width=170 * mm, height=65 * mm):
        super().__init__()
        self.measured = measured
        self.predicted = predicted
        self.width = width
        self.height = height

    def draw(self):
        canvas = self.canv
        left, bottom = 17 * mm, 12 * mm
        plot_w, plot_h = self.width - 29 * mm, self.height - 22 * mm
        budgets = [0, 5, 10, 25]
        xs = [left + plot_w * index / 3 for index in range(4)]
        canvas.setFont("Sans", 6.5)
        for value in range(0, 101, 20):
            y = bottom + plot_h * value / 100
            canvas.setStrokeColor(LIGHT)
            canvas.line(left, y, left + plot_w, y)
            canvas.setFillColor(DARK)
            canvas.drawRightString(left - 2 * mm, y - 2, str(value))
        for x, budget in zip(xs, budgets, strict=True):
            canvas.drawCentredString(x, bottom - 4 * mm, str(budget))
        canvas.drawCentredString(left + plot_w / 2, 1 * mm, "Target demonstrations")
        canvas.saveState()
        canvas.translate(3 * mm, bottom + plot_h / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, "Success, %")
        canvas.restoreState()

        rows = {(row["method"], int(row["budget"])): row for row in self.measured}
        zero = rows[("zero", 0)]
        styles = [
            ("baseline", [], False),
            ("method", [5, 2], True),
        ]
        for method, dash, square in styles:
            values = [zero] + [rows[(method, budget)] for budget in budgets[1:]]
            points = []
            for x, row in zip(xs, values, strict=True):
                rate = 100 * float(row["success_rate"])
                lo = 100 * float(row["ci_low"])
                hi = 100 * float(row["ci_high"])
                y = bottom + plot_h * rate / 100
                y_lo = bottom + plot_h * lo / 100
                y_hi = bottom + plot_h * hi / 100
                canvas.setStrokeColor(BLACK)
                canvas.setLineWidth(0.65)
                canvas.line(x, y_lo, x, y_hi)
                canvas.line(x - 1.5 * mm, y_lo, x + 1.5 * mm, y_lo)
                canvas.line(x - 1.5 * mm, y_hi, x + 1.5 * mm, y_hi)
                points.append((x, y))
            canvas.setLineWidth(1.5)
            canvas.setDash(*dash) if dash else canvas.setDash()
            canvas.lines([(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]) for i in range(3)])
            canvas.setDash()
            for x, y in points:
                if square:
                    canvas.rect(x - 2, y - 2, 4, 4, stroke=1, fill=0)
                else:
                    canvas.circle(x, y, 2, stroke=1, fill=1)

        # Preregistered expectations are shown as faint dotted references.
        zero_pred = self.predicted["zero_shot_percent"]
        for key in ("baseline_percent", "method_percent"):
            values = [zero_pred] + [self.predicted[key][budget] for budget in budgets[1:]]
            points = [(x, bottom + plot_h * value / 100) for x, value in zip(xs, values, strict=True)]
            canvas.setStrokeColor(MID)
            canvas.setLineWidth(0.55)
            canvas.setDash(1, 2)
            canvas.lines([(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]) for i in range(3)])
        canvas.setDash()
        canvas.setFillColor(BLACK)
        canvas.setFont("Sans", 6.3)
        canvas.circle(left + 93 * mm, self.height - 7 * mm, 2, stroke=1, fill=1)
        canvas.drawString(left + 97 * mm, self.height - 9 * mm, "baseline measured")
        canvas.rect(left + 93 * mm - 2, self.height - 13 * mm - 2, 4, 4, stroke=1, fill=0)
        canvas.drawString(left + 97 * mm, self.height - 15 * mm, "LoRA+aug measured")
        canvas.setStrokeColor(MID)
        canvas.setDash(1, 2)
        canvas.line(left + 90 * mm, self.height - 19 * mm, left + 96 * mm, self.height - 19 * mm)
        canvas.setDash()
        canvas.drawString(left + 97 * mm, self.height - 21 * mm, "preregistered prediction")


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def grayscale_failures(catalog: list[dict]) -> list[Path]:
    TMP.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, case in enumerate(catalog, start=1):
        source = Path(case["figure"])
        if not source.is_absolute():
            source = ROOT / source
        destination = TMP / f"failure-{index}-gray.png"
        with PILImage.open(source) as image:
            image.convert("L").save(destination)
        outputs.append(destination)
    return outputs


def main() -> None:
    register_fonts()
    style = make_styles()
    summary = read_csv(RESULTS / "cost_curve.csv")
    per_run = read_csv(RESULTS / "per_run.csv")
    language = read_csv(RESULTS / "language_control.csv")
    failure_path = RESULTS / "failures/catalog.json"
    bonus_path = RESULTS / "reward_ranking.json"
    if not failure_path.exists() or not bonus_path.exists():
        raise FileNotFoundError("Failure catalog and Bonus B ranking must exist before report export")
    failures = json.loads(failure_path.read_text(encoding="utf-8"))
    bonus = json.loads(bonus_path.read_text(encoding="utf-8"))
    predictions = read_predictions(ROOT / "predictions.yaml")
    manifest = json.loads((ROOT / "artifacts/protocol_manifest.json").read_text(encoding="utf-8"))
    gray_failures = grayscale_failures(failures)

    rows = {(row["method"], int(row["budget"])): row for row in summary}
    gains = {
        budget: 100 * (
            float(rows[("method", budget)]["success_rate"])
            - float(rows[("baseline", budget)]["success_rate"])
        )
        for budget in (5, 10, 25)
    }
    correct = sum(float(row["correct_rate"]) for row in language) / len(language)
    wrong = sum(float(row["wrong_rate"]) for row in language) / len(language)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=14 * mm,
        title="SmolVLA low-demo adaptation on LIBERO",
        author="Reproducible experiment report",
    )
    story = []

    story += [
        Spacer(1, 2 * mm),
        Paragraph("Low-demo adaptation of SmolVLA on LIBERO", style["title"]),
        Paragraph(
            "A leakage-controlled, compute-limited study of 0/5/10/25 demonstrations",
            style["subtitle"],
        ),
        Paragraph("Abstract", style["h1"]),
        Paragraph(
            "We adapt a 450M-parameter SmolVLA policy to the first three LIBERO-Goal tasks. "
            "The experiment preserves the requested task set, demonstration budgets, two training "
            "seeds and 20 simulator episodes per task/checkpoint. Compute is limited only through "
            "5,000 seen-pretraining updates and 500 target-adaptation updates. We compare ordinary "
            "fine-tuning with rank-16 LoRA plus mild image augmentation, include a paired wrong-language "
            f"control, analyze three real failures, and rank checkpoints without reward. The measured "
            f"LoRA-minus-baseline differences at 5/10/25 demonstrations are {gains[5]:+.1f}, {gains[10]:+.1f}, and "
            f"{gains[25]:+.1f} percentage points. Only the naive 10-demo condition succeeds (8/120); "
            "the proposed method records no success at any budget.",
            style["abstract"],
        ),
        Paragraph("1. Experimental design", style["h1"]),
        Paragraph(
            "The seen pool is the available `libero_90` conversion: 73 task ids, 3,921 episodes and "
            "569,249 frames. Targets are simulator task ids 0-2 in `libero_goal`. The simulator and "
            "dataset use different task orders, so demonstrations are joined by exact canonical prompt, "
            "not numeric task id. This prevents silent training on the wrong target.",
            style["body"],
        ),
        table(
            [
                ["Simulator id", "Dataset task_index", "Instruction", "First five episode ids"],
                ["0", "9", "open the middle drawer of the cabinet", "20, 26, 31, 42, 58"],
                ["1", "7", "put the bowl on the stove", "13, 15, 16, 22, 36"],
                ["2", "4", "put the wine bottle on top of the cabinet", "6, 10, 17, 18, 25"],
            ],
            [22 * mm, 30 * mm, 73 * mm, 45 * mm],
        ),
        Spacer(1, 3 * mm),
        Paragraph("Training and evaluation", style["h2"]),
        table(
            [
                ["Component", "Fixed specification"],
                ["Base / seen", "lerobot/smolvla_base; 5,000 updates x seeds 17, 29 on all 73 seen tasks"],
                ["Naive baseline", "500 updates; standard action-expert fine-tuning; no transforms"],
                ["Proposed method", "rank-16 LoRA; alpha 16; at most two random mild image transforms"],
                ["Evaluation", "300-step LIBERO horizon; 20 episodes per task/checkpoint; seed 1000"],
                ["Hardware", "RTX 4070 Ti SUPER 16 GB; PyTorch 2.7.1+cu126; LeRobot 9e145849..."],
            ],
            [39 * mm, 131 * mm],
        ),
        Spacer(1, 3 * mm),
        Paragraph("Preregistration", style["h2"]),
        Paragraph(
            "Before adaptation runs we predicted a monotone curve and the largest LoRA advantage at "
            "five demonstrations. Prediction SHA-256 and exact nested episode prefixes were fixed before "
            f"training; manifest digest: {manifest['sha256'][:24]}... .",
            style["body"],
        ),
        Paragraph(
            "The public 40-task LIBERO checkpoint was rejected because its training universe overlaps "
            "the held-out goal tasks. All measured target checkpoints descend from our own seen run.",
            style["body"],
        ),
        PageBreak(),
    ]

    curve_table = [["Demos", "Zero-shot", "Naive", "LoRA+aug", "Difference"]]
    for budget in (0, 5, 10, 25):
        if budget == 0:
            zero_row = rows[("zero", 0)]
            curve_table.append(["0", percent(float(zero_row["success_rate"])), "-", "-", "-"])
        else:
            base = rows[("baseline", budget)]
            method = rows[("method", budget)]
            curve_table.append(
                [
                    str(budget),
                    "-",
                    percent(float(base["success_rate"])),
                    percent(float(method["success_rate"])),
                    f"{gains[budget]:+.1f} pp",
                ]
            )
    story += [
        Paragraph("2. Results", style["h1"]),
        Paragraph("Cost curve", style["h2"]),
        CostCurve(summary, predictions["expected_shape"]),
        table(curve_table, [26 * mm, 35 * mm, 35 * mm, 38 * mm, 36 * mm]),
        Spacer(1, 3 * mm),
        Paragraph(
            "Markers and 95% Wilson intervals are measured simulator outcomes; faint dotted lines are "
            "the preregistered expectations. The shared zero-shot point is evaluated from both seen seeds. "
            "No incomplete run is imputed.",
            style["small"],
        ),
        Paragraph("Language control", style["h2"]),
        Paragraph(
            f"With the correct instruction, mean zero-shot success across task/seed cells is "
            f"{percent(correct)}; replacing only the instruction with another target task gives "
            f"{percent(wrong)} (drop {100 * (correct - wrong):+.1f} pp). The paired reset seed and "
            "environment task are unchanged. Because both conditions are at the performance floor, this "
            "control is uninformative about language reliance; it does not establish language invariance.",
            style["body"],
        ),
        table(
            [["Task", "Seed", "Correct prompt", "Wrong prompt", "Drop"]]
            + [
                [
                    row["task_id"], row["seed"], percent(float(row["correct_rate"])),
                    percent(float(row["wrong_rate"])), f"{100 * float(row['drop']):+.1f} pp",
                ]
                for row in language
            ],
            [25 * mm, 25 * mm, 42 * mm, 42 * mm, 36 * mm],
            font_size=6.6,
        ),
        Spacer(1, 3 * mm),
        Paragraph("Per-task variation (pooled over two training seeds)", style["h2"]),
        table(
            [["Demos", "Task", "Naive (n=40)", "LoRA+aug (n=40)", "Difference"]]
            + [
                [
                    str(budget),
                    str(task_id),
                    percent(
                        sum(
                            float(row["success_rate"])
                            for row in per_run
                            if row["stage"] == "baseline"
                            and int(row["budget"]) == budget
                            and int(row["task_id"]) == task_id
                        )
                        / 2
                    ),
                    percent(
                        sum(
                            float(row["success_rate"])
                            for row in per_run
                            if row["stage"] == "method"
                            and int(row["budget"]) == budget
                            and int(row["task_id"]) == task_id
                        )
                        / 2
                    ),
                    f"{100 * (sum(float(row['success_rate']) for row in per_run if row['stage'] == 'method' and int(row['budget']) == budget and int(row['task_id']) == task_id) - sum(float(row['success_rate']) for row in per_run if row['stage'] == 'baseline' and int(row['budget']) == budget and int(row['task_id']) == task_id)) / 2:+.1f} pp",
                ]
                for budget in (5, 10, 25)
                for task_id in (0, 1, 2)
            ],
            [27 * mm, 27 * mm, 40 * mm, 43 * mm, 33 * mm],
            font_size=6.3,
        ),
        PageBreak(),
    ]

    failure_hypotheses = [
        (
            "Grounding versus control",
            "The drawer remains closed while the end effector traverses the central workspace without contact.",
            (
                "Hold state and instruction fixed while changing only target texture/location. Consistent "
                "motion toward the wrong object supports grounding failure; correct approach followed by a "
                "contact miss supports low-level control failure."
            ),
        ),
        (
            "Action-chunk overshoot",
            "The arm reaches the stove region, but no completed bowl placement is visible at the horizon.",
            (
                "Replay the same initial states with n_action_steps in {1, 5, 10}. Improvement restricted to "
                "contact phases under shorter execution supports chunking; an unchanged wrong trajectory does not."
            ),
        ),
        (
            "Stage transition",
            "The bottle remains stationary while the arm moves toward the upper-right scene boundary.",
            (
                "Initialize immediately before and after a subgoal boundary; compare the full instruction with "
                "an explicit remaining-step prompt. Improvement only under the latter supports planning/language failure."
            ),
        ),
    ]
    story += [Paragraph("3. Failure analysis", style["h1"])]
    for index, (case, image_path, hypothesis) in enumerate(
        zip(failures, gray_failures, failure_hypotheses, strict=True), start=1
    ):
        details = (
            f"<b>Failure {index}.</b> LoRA+aug, {case['budget']} demonstrations, task "
            f"{case['task_id']}, seed {case['seed']}, episode {case['episode_index']}. "
            f"<b>Observed.</b> {hypothesis[1]} "
            f"<b>Hypothesis test - {hypothesis[0]}.</b> {hypothesis[2]}"
        )
        row = Table(
            [[Image(str(image_path), width=88 * mm, height=29.3 * mm), Paragraph(details, style["body"])]],
            colWidths=[92 * mm, 78 * mm],
        )
        row.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOX", (0, 0), (-1, -1), 0.4, MID),
                    ("LINEAFTER", (0, 0), (0, 0), 0.4, MID),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story += [row, Spacer(1, 3 * mm)]
    story += [
        Paragraph(
            "Frames are sampled from actual 300-step simulator MP4s. The catalog preserves method, "
            "budget, task, training seed, episode id and source video path. These experiments are proposed "
            "diagnostic interventions; they are not retroactively treated as performed evidence.",
            style["small"],
        ),
        PageBreak(),
    ]

    story += [
        Paragraph("4. Reward-free evaluation and discussion", style["h1"]),
        Paragraph("Bonus B: checkpoint ranking without simulator reward", style["h2"]),
        Paragraph(
            "The small critic is trained from scratch on RGB frames from seen demonstrations using only "
            "normalized temporal position; action values are never read. Both critics receive the same "
            "deterministic five-rollout sample per checkpoint; Robometer-4B also receives task text. Their "
            "scores rank checkpoints and are compared with simulator success using Spearman correlation.",
            style["body"],
        ),
        table(
            [
                ["Evaluator", "Checkpoints", "Spearman rho vs success"],
                ["Small temporal CNN", str(bonus["n_checkpoints"]), f"{bonus['small_spearman']:.3f}"],
                ["Robometer-4B-LIBERO", str(bonus["n_checkpoints"]), f"{bonus['robometer_spearman']:.3f}"],
            ],
            [62 * mm, 42 * mm, 66 * mm],
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "Direct optimization against either critic is unsafe. The small model can exploit temporal "
            "shortcuts and terminal-looking poses; Robometer can be miscalibrated out of distribution. "
            "We therefore predefine reward hacking as critic improvement accompanied by lower simulator "
            "success on a frozen audit set, and do not mix critic optimization into the main cost curve.",
            style["body"],
        ),
        Paragraph("Predictions versus results", style["h2"]),
        Paragraph(
            f"We predicted the largest LoRA advantage at five demonstrations. The measured differences "
            f"were {gains[5]:+.1f}/{gains[10]:+.1f}/{gains[25]:+.1f} pp at 5/10/25. "
            "The result is interpreted from the full per-task/per-seed table rather than only the pooled mean; "
            "two seeds remain insufficient for small-effect claims.",
            style["body"],
        ),
        Paragraph("Rejected approaches", style["h2"]),
        table(
            [
                ["Approach", "Reason for rejection"],
                ["Full VLM fine-tuning", "Poorly identified by 5-25 demos and incompatible with a matched 16 GB budget."],
                ["Mixing other held-out tasks", "Would improve the headline number through direct target-demo leakage."],
                ["Best-demo selection", "Adds unreported manual supervision and breaks the fixed-prefix budget."],
                ["Open 40-task checkpoint", "Its training universe overlaps the held-out goal tasks."],
            ],
            [47 * mm, 123 * mm],
        ),
        Spacer(1, 3 * mm),
        Paragraph("Most unexpected observation", style["h2"]),
        Paragraph(
            "More target data did not help: naive fine-tuning rose to 8/120 at ten demonstrations and "
            "returned to 0/120 at 25, while LoRA remained at zero. The result points to update-budget and "
            "optimization mismatch rather than simple demo scarcity. Separately, the simulator and data "
            "conversion disagree on task ordering, so prompt-based identity checks were indispensable.",
            style["body"],
        ),
        Paragraph("Reproducibility", style["h2"]),
        Paragraph(
            "Exact commands, immutable episode prefixes, raw eval_info.json files, 960 outcomes and 480 "
            "retained MP4s, checkpoint metadata and aggregation tables are preserved. Sources: SmolVLA "
            "(arXiv:2506.01844), LIBERO (arXiv:2306.03310), LeRobot LIBERO documentation and Robometer.",
            style["small"],
        ),
    ]

    doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    print(OUTPUT)


if __name__ == "__main__":
    main()
