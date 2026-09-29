"""Build the BME mid-term project proposal PDF."""

from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
pdfmetrics.registerFont(TTFont("TNR", str(FONT_DIR / "Times New Roman.ttf")))
pdfmetrics.registerFont(TTFont("TNR-Bold", str(FONT_DIR / "Times New Roman Bold.ttf")))
pdfmetrics.registerFont(
    TTFont("TNR-Italic", str(FONT_DIR / "Times New Roman Italic.ttf"))
)
pdfmetrics.registerFont(
    TTFont("TNR-BoldItalic", str(FONT_DIR / "Times New Roman Bold Italic.ttf"))
)
registerFontFamily(
    "TNR",
    normal="TNR",
    bold="TNR-Bold",
    italic="TNR-Italic",
    boldItalic="TNR-BoldItalic",
)

OUT = Path(__file__).with_name("midterm-report.pdf")


def styles():
    base = getSampleStyleSheet()
    leading = 11.5
    common = dict(fontSize=10, leading=leading, textColor="black")
    return {
        "title": ParagraphStyle(
            "Title10",
            parent=base["Normal"],
            **common,
            fontName="TNR-Bold",
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle10",
            parent=base["Normal"],
            **common,
            fontName="TNR",
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "h": ParagraphStyle(
            "H10",
            parent=base["Normal"],
            **common,
            fontName="TNR-Bold",
            alignment=TA_LEFT,
            spaceBefore=5,
            spaceAfter=1,
        ),
        "body": ParagraphStyle(
            "Body10",
            parent=base["Normal"],
            **common,
            fontName="TNR",
            alignment=TA_JUSTIFY,
            spaceAfter=2,
        ),
        "q": ParagraphStyle(
            "Q10",
            parent=base["Normal"],
            **common,
            fontName="TNR",
            alignment=TA_JUSTIFY,
            spaceAfter=2,
            leftIndent=10,
            firstLineIndent=-10,
        ),
    }


def q(prompt: str, answer: str, s) -> Paragraph:
    return Paragraph(f"<b>{prompt}</b> {answer}", s["q"])


def build() -> None:
    s = styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title="Mid-Term Report: 12-Lead ECG Superdiagnosis under Structured Missing Leads",
        author="Artur Lima, Pedro Nogueira",
    )
    story = [
        Paragraph("Mid-Term Report", s["title"]),
        Paragraph("Artur Lima, Pedro Nogueira &nbsp;|&nbsp; 18 September 2026", s["subtitle"]),
        Paragraph("1. Project Information", s["h"]),
        q("Provisional project title:",
          "Classification of Diagnostic Superclasses from 12-Lead ECGs "
          "under Structured Missing-Lead Conditions.", s),
        q("Names of all group members:", "Artur Lima, Pedro Nogueira.", s),
        Paragraph("2. Data Collection", s["h"]),
        q(
            "How will you collect your data?",
            "We are not recruiting patients or recording new ECGs. The source is "
            "PTB-XL v1.0.3 on PhysioNet: 21,799 clinical 12-lead ECGs from 18,869 "
            "patients. Waveforms were collected on Schiller AG devices between "
            "October 1989 and June 1996 and later curated at the "
            "Physikalisch-Technische Bundesanstalt (PTB). Each record is 10 seconds. "
            "Up to two cardiologists assigned SCP-ECG statements. We use the "
            "published 100 Hz waveforms (1,000 samples per lead), not the 500 Hz "
            "files.",
            s,
        ),
        q(
            "What procedure or steps will you follow?",
            "Download <font name='TNR-Italic'>ptbxl_database.csv</font>, "
            "<font name='TNR-Italic'>scp_statements.csv</font>, and every "
            "<font name='TNR-Italic'>records100</font> .dat/.hea pair from PhysioNet. "
            "Keep the official <font name='TNR-Italic'>strat_fold</font> split: folds "
            "1–8 train (17,418 records, 15,023 patients), fold 9 validation (2,183), "
            "fold 10 test (2,198). PhysioNet marks folds 9 and 10 as fully "
            "human-validated and recommends 1–8 / 9 / 10 as train / validation / "
            "test. Patient IDs do not cross those three sets; that check already "
            "passed. The 21,799 100 Hz records are already on disk.",
            s,
        ),
        q(
            "What equipment or measurement tools will you use?",
            "The original equipment was Schiller 12-lead ECG machines. PhysioNet "
            "releases the traces in WFDB at 500 Hz (16-bit, 1 µV/LSB) and as a "
            "100 Hz downsample; we use only the latter. We do not add sensors. "
            "Analysis is in Python (WFDB, NumPy, pandas, scikit-learn, PyTorch) on "
            "a laptop, with Colab only if training needs a GPU.",
            s,
        ),
        q(
            "Will you need to clean, filter, or process the raw data before analysis?",
            "Yes. Diagnostic SCP codes are mapped through "
            "<font name='TNR-Italic'>scp_statements.csv</font> onto five superclasses: "
            "NORM, MI (myocardial infarction), STTC (ST/T change), CD (conduction "
            "disturbance), and HYP (hypertrophy). The 411 records with no diagnostic "
            "superclass stay in the set as all-zero labels. Diagnostic statements "
            "are counted from the SCP code keys, including unknown likelihood "
            "(coded 0); that matches PhysioNet’s published superclass counts and "
            "their example aggregation script. Each "
            "lead is z-scored with mean and standard deviation from the training folds "
            "only, so the test set does not leak into the scale. Age and sex are in "
            "the metadata and will not be model inputs. At evaluation we zero a few "
            "fixed lead sets (not random subsets) to imitate a missing electrode. We "
            "do not use random lead dropout during training.",
            s,
        ),
        q(
            "Approximately how much data will you collect?",
            "The full public 100 Hz corpus: 21,799 records × 12 leads × 1,000 samples "
            "(about 1.0 GB packed as float32). That is the whole dataset at this "
            "rate, not a subsample.",
            s,
        ),
        Paragraph("3. Data Set", s["h"]),
        q(
            "What will each data point in your data set represent?",
            "One 10-second 12-lead ECG (<font name='TNR-Italic'>ecg_id</font>) from "
            "one patient (<font name='TNR-Italic'>patient_id</font>). Some patients "
            "have more than one ECG; the official folds already keep a patient’s "
            "records in a single fold.",
            s,
        ),
        q(
            "What variables will you measure?",
            "Millivolt time series on leads I, II, III, aVR, aVL, aVF, V1–V6; the "
            "five superclass labels; official fold; patient id. Full-set class counts "
            "are NORM 9,514, MI 5,469, STTC 5,235, CD 4,898, HYP 2,649. They add to "
            "more than 21,799 because labels co-occur (5,144 records have two or more "
            "superclasses). Training positive rates are 0.436, 0.251, 0.240, 0.224, "
            "and 0.122.",
            s,
        ),
        q(
            "Which variables will be features (inputs)?",
            "The 12 × 1,000 voltage samples (12,000 numerical values). At test time a "
            "12-dimensional lead mask records which of three fixed conditions is "
            "applied: all leads, V1–V3 off, or I and II only. Patient id is not a "
            "feature.",
            s,
        ),
        q(
            "Which variable(s) will be the response (output)?",
            "Five binary superclass indicators. This is multi-label, not a single "
            "five-way class: one ECG can be MI and CD at the same time.",
            s,
        ),
        q(
            "How many variables will you have?",
            "12,000 input voltage values and 5 output labels. Identifiers (ecg_id, "
            "patient_id, fold) are stored for splitting and the demo, not as predictors.",
            s,
        ),
        q(
            "What type of data are the variables?",
            "Voltages are numerical/interval (millivolts). Superclass labels are "
            "categorical/nominal, stored as 0/1. Lead name is categorical. Fold and "
            "patient id are identifiers.",
            s,
        ),
        Paragraph("4. Data Analysis", s["h"]),
        q(
            "What relationships or patterns do you expect to find?",
            "A model that sees the waveform should beat a constant predictor that "
            "only knows training prevalence. That baseline is already computed on "
            "fold 10: every class has AUROC 0.5, and macro-AUPRC is 0.254. HYP should "
            "be the weakest class because it is the rarest. The comparison we care "
            "about is across three fixed test conditions, reported as separate rows: "
            "all 12 leads; V1–V3 off (precordial leads used for anterior/septal "
            "infarct patterns); "
            "and I+II only. Limb leads can reconstruct III and the augmented leads "
            "from Einthoven’s relations, so a small drop on I+II is not the same "
            "finding as a drop when V1–V3 are gone.",
            s,
        ),
        q(
            "What type of model or analysis do you plan to use?",
            "The analysis is a small missing-lead comparison table on the official "
            "PTB-XL split: every model we train is scored on the same three test "
            "conditions. The first model is a 1D convolutional network on the "
            "(12, 1000) tensor, trained with binary cross-entropy with logits and "
            "early-stopped on fold-9 macro-AUROC. If that run is stable, one small "
            "supervised Transformer on the same split and the same table. No "
            "self-supervised pretraining. This is not a diagnostic device.",
            s,
        ),
        q(
            "Why is that model appropriate for your type of data?",
            "ECG diagnosis is about local waveform shape (P, QRS, ST) along each lead. "
            "A 1D CNN looks for those shapes in a multivariate numerical time series. "
            "A Transformer is a second architecture that can mix information across "
            "time and leads. Both sit on 12,000 interval-valued inputs and five "
            "binary outputs, which matches the PTB-XL superdiagnosis task used in "
            "published CNN benchmarks on this dataset.",
            s,
        ),
        q(
            "How will you evaluate how well your model works?",
            "Fold 10 only, after model selection on fold 9. Primary metrics: per-class "
            "AUROC and AUPRC, plus their macros. The CNN has to beat the 0.5 "
            "macro-AUROC floor on clean 12-lead. Then the same metrics, without "
            "retraining, on the two missing-lead rows (V1–V3 off; I+II only). Every "
            "trained model fills the same three rows so the table is a small "
            "benchmark, not a one-off plot. Rows stay separate; they are not averaged "
            "into one robustness score. Because a record can carry several labels, a "
            "single 5×5 confusion matrix is the wrong summary. If we show confusion, "
            "it will be five binary tables at thresholds tuned on fold 9.",
            s,
        ),
        q(
            "How will you visualize your results?",
            "The main figure is the three-row missing-lead table (clean, V1–V3 off, "
            "I+II only) with per-class AUROC, including HYP. Also: a grid of example "
            "lead-II traces (already made for a 100-record audit), training curves, "
            "and bars against the prevalence baseline. Last, a small demo: pick a "
            "test ECG, mute the same lead sets, and watch the five class scores change.",
            s,
        ),
        Paragraph(
            "The deliverable is that comparison table on fold 10. The models are not "
            "a clinical ECG interpretation system.",
            s["body"],
        ),
    ]
    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
