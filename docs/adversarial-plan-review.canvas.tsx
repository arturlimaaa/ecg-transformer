import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  CollapsibleSection,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Link,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type Severity = "blocker" | "high" | "medium";
type Filter = "all" | Severity;

const KILL_SHOTS = [
  {
    id: "einthoven",
    title: "Lead reconstruction is often algebra, not learning",
    body: "Leads I, II, III and aVR, aVL, aVF are linearly dependent. III = II − I. aVR = −(I+II)/2. Masking one to three limb leads and asking for MSE reconstruction lets the model copy Einthoven’s law. That does not prove a useful representation for MI, STTC, or HYP, which depend on precordial morphology. The plan’s example (mask V1, V2) is the hard case. The default protocol (any one-to-three leads) is not.",
  },
  {
    id: "control",
    title: "The robustness claim has no control",
    body: "The plan optionally drops leads during supervised training, then treats masked-lead pretraining as the cause of robustness. If dropout is on, SSL is confounded with augmentation. If dropout is off, the CNN never saw missing leads, so SSL wins by construction. The missing cell is a Transformer (and CNN) trained with the same lead masks as the SSL model, with no reconstruction head.",
  },
  {
    id: "prior-art",
    title: "The written question is already answered",
    body: "MaeFE (2022) already pretrains by reconstructing masked leads (MLAE). Oh et al. (2022) already use random lead masking for reduced-lead robustness on PTB-XL. ST-MEM (2024) already compares time masking vs lead masking with lead embeddings. TolerantECG (2025) already benchmarks PTB-XL superdiagnosis under missing leads and noise. A class project can reproduce a slice of that. It cannot claim the question as new.",
  },
] as const;

const PRIOR_ART = [
  [
    "MaeFE MLAE",
    "Zhang et al., IEEE TIM 2022",
    "Masked-lead autoencoder. Exact pretext in this plan.",
    <Link href="https://doi.org/10.1109/tim.2022.3228267">doi</Link>,
  ],
  [
    "Random Lead Masking",
    "Oh et al., ML4H 2022",
    "Zero each lead with p=0.5. Reduced-lead robustness on PTB-XL.",
    <Link href="https://proceedings.mlr.press/v174/oh22a.html">PMLR</Link>,
  ],
  [
    "ST-MEM",
    "Na et al., 2024",
    "Spatio-temporal MAE. Compares MTAE vs MLAE. Lead embeddings. Reduced-lead fine-tune.",
    <Link href="https://arxiv.org/abs/2402.09450">arXiv</Link>,
  ],
  [
    "TolerantECG",
    "Nguyen et al., ACM MM 2025",
    "Missing-lead and noise robustness on PTB-XL superdiag. Beats ECG-FM in several settings.",
    <Link href="https://arxiv.org/abs/2507.09887">arXiv</Link>,
  ],
  [
    "PTB-XL CNN bench",
    "Strodthoff et al., JBHI 2020",
    "resnet1d_wang macro-AUROC 0.930 on superdiagnosis. This is the CNN to beat, not a toy conv net.",
    <Link href="https://github.com/helme/ecg_ptbxl_benchmarking">code</Link>,
  ],
  [
    "ECG SSL on PTB-XL",
    "Mehari and Strodthoff, 2022",
    "SimCLR / BYOL / CPC on 12-lead ECG. Already tests label efficiency.",
    <Link href="https://doi.org/10.1016/j.compbiomed.2021.105114">doi</Link>,
  ],
];

const PROTOCOL: Array<{
  severity: Severity;
  issue: string;
  why: string;
  fix: string;
}> = [
  {
    severity: "blocker",
    issue: "SSL on the same 17k labeled records",
    why: "Pretraining then fine-tuning on PTB-XL train folds is a pretext task, not unlabeled-data SSL. There is no extra corpus.",
    fix: "Either pretrain on MIMIC-IV-ECG / PhysioNet 2021 and fine-tune PTB-XL, or drop the SSL framing and call it a pretext ablation.",
  },
  {
    severity: "blocker",
    issue: "Zeros as a mask",
    why: "After z-score, zero is the training mean. A missing lead looks like a perfectly average ECG, not an absent channel.",
    fix: "Learned mask tokens only. Feed the lead_mask into the encoder. Do not treat zeros as equivalent.",
  },
  {
    severity: "blocker",
    issue: "Random 1 or 3 missing leads",
    why: "Clinical missingness is structured: limb-only, I+II+V2, smartwatch single-lead, loose V lead. Random subsets average easy Einthoven cases with hard precordial cases.",
    fix: "Freeze three test masks: {I,II}, limb-6, and three precordial. Report each. Never average them into one number.",
  },
  {
    severity: "high",
    issue: "Relative drop as the headline metric",
    why: "corrupted / clean is not a robustness score. A weaker clean model inflates the ratio even when corrupted AUROC is worse.",
    fix: "Report absolute macro-AUROC and macro-AUPRC on each condition, plus the delta. Never lead with the ratio.",
  },
  {
    severity: "high",
    issue: "Toy CNN as the deep baseline",
    why: "Published 1D ResNets sit at ~0.93 macro-AUROC. A small conv net at 0.88 makes any Transformer look good.",
    fix: "Implement resnet1d_wang or xresnet1d101 from the official PTB-XL benchmark and match their fold-10 number before claiming a new model.",
  },
  {
    severity: "high",
    issue: "100 Hz patches of 40 samples",
    why: "400 ms patches mix P, QRS, and ST. Patch transformers on PTB-XL already improve from 0.856 at 100 Hz to 0.877 at 500 Hz, while CNNs do not.",
    fix: "If the Transformer is the scientific object, use 500 Hz or patches ≤ 80 ms (8 samples at 100 Hz). State the choice as a limitation if you keep 100 Hz.",
  },
  {
    severity: "high",
    issue: "macro-F1 with no operating point",
    why: "F1 on imbalanced multi-label data is a function of the threshold. Accuracy is correctly rejected. F1 without a rule is the same trap.",
    fix: "Tune per-class thresholds on fold 9. Report fold 10 F1 at those thresholds. Keep AUROC/AUPRC as primary.",
  },
  {
    severity: "high",
    issue: "No bootstrap, one test fold",
    why: "Strodthoff reports AUROC with bootstrap CIs on fold 10. A 0.01 gap on 2.2k test records is often noise, especially for HYP (n≈265 positives).",
    fix: "Bootstrap the test set. Do not claim a winner inside overlapping CIs.",
  },
  {
    severity: "high",
    issue: "Unspecified noise",
    why: "Gaussian noise is not electrode artifact. PTB-XL already tags static_noise, burst_noise, baseline_drift, and electrodes_problems.",
    fix: "Set SNR in dB. Add MIT-BIH NST baseline wander / muscle / electrode motion. Also slice the real PTB-XL artifact flags.",
  },
  {
    severity: "high",
    issue: "Fine-tune can erase the pretext",
    why: "Full fine-tuning on clean 12-lead labels often destroys missing-lead behavior unless masks continue at fine-tune time.",
    fix: "Keep lead masking during fine-tune, or freeze the encoder and train a linear head as a probe. Report both.",
  },
  {
    severity: "medium",
    issue: "HYP and empty superclasses",
    why: "HYP has 2649 / 21799 records. Some records have no diagnostic superclass. Unweighted BCE will ignore HYP. Empty-label rows will look like all-negative.",
    fix: "Decide whether to keep unlabeled-diagnostic records. Use pos_weight or per-class loss weights. Report HYP separately always.",
  },
  {
    severity: "medium",
    issue: "SCP likelihood dumped into binary labels",
    why: "scp_codes carry 15 / 50 / 80 / 100 likelihoods. Treating all as positive is the common choice. It is still a labeling decision.",
    fix: "Document the aggregation. Optionally ablate certain (100) vs all statements.",
  },
  {
    severity: "medium",
    issue: "Interpretability theater in the writeup",
    why: "Occlusion maps and t-SNE on a 5-way multi-label embedding do not test the hypothesis. Attention is not a medical explanation. The plan already knows this and still budgets the figures.",
    fix: "One occlusion example for a known lead-local class (anterior MI, V leads). Skip UMAP unless a reviewer asks.",
  },
];

const KEEP = [
  [
    "Official folds 1–8 / 9 / 10",
    "Patient-aware, published, comparable. Do not invent a random split.",
  ],
  [
    "Five-class multi-label + BCEWithLogits",
    "Matches the PTB-XL superdiagnosis task. Correct loss.",
  ],
  [
    "Train-only normalization",
    "The leak warning is right. Keep it.",
  ],
  [
    "Synthetic smoke test before PTB-XL",
    "Correct debugging order for a first PyTorch project.",
  ],
  [
    "MVP without SSL",
    "Right risk control. The MVP is a methods tutorial, not a paper. Name it that way.",
  ],
  [
    "First milestone: 100 records, labels, no patient leak",
    "Correct first oracle. That is M2. Repo chrome is not a milestone.",
  ],
];

function severityTone(s: Severity): "danger" | "warning" | "info" {
  if (s === "blocker") return "danger";
  if (s === "high") return "warning";
  return "info";
}

function severityLabel(s: Severity): string {
  if (s === "blocker") return "Blocker";
  if (s === "high") return "High";
  return "Medium";
}

export default function AdversarialPlanReview() {
  const theme = useHostTheme();
  const [filter, setFilter] = useCanvasState<Filter>("severity-filter", "all");

  const protocolRows = PROTOCOL.filter(
    (row) => filter === "all" || row.severity === filter,
  );

  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>ECG Transformer: coursework cut</H1>
        <Text tone="secondary">
          Locked 12 Sep 2026: finishable class project with numbers and a
          demo. Not a research paper. SSL and the protocol rewrite stay
          parked below.
        </Text>
      </Stack>

      <Callout tone="success" title="Commitment">
        Train a CNN and a small supervised Transformer on PTB-XL
        superdiagnosis. Show that predictions change when leads are dropped.
        Write it as a student benchmark, not a clinical system and not a
        claim that masked-lead pretraining is new.
      </Callout>

      <Row gap={24} wrap>
        <Stat value="8" label="Milestones in sequence" />
        <Stat value="M6" label="Only skippable (Transformer)" tone="warning" />
        <Stat value="M8" label="Demo is the product" tone="success" />
        <Stat value="Cut" label="SSL stays parked" />
      </Row>

      <H2>Milestones</H2>
      <Text>
        One open at a time. A milestone is done only when the oracle passed
        and you read the output. Calendar time is irrelevant.
      </Text>
      <Table
        headers={["", "Milestone", "Oracle", "If it fails"]}
        rows={[
          [
            "M1",
            "Synthetic smoke",
            "One conv layer on fake (12, 1000) batches. Loss is finite and falls over 20 steps.",
            "Do not touch PTB-XL. Fix the train loop.",
          ],
          [
            "M2",
            "100-record audit",
            "100 PTB-XL 100 Hz records plotted. Labels match scp superclasses. Patient overlap across official folds is 0.",
            "Fix the loader. Do not train.",
          ],
          [
            "M3",
            "Full pipeline",
            "Dataset returns signal (12, 1000), label (5,), lead_mask (12,). Folds 1–8 / 9 / 10. Train/val/test patient sets disjoint. Norm stats from train only.",
            "Stay here. A wrong split poisons every later number.",
          ],
          [
            "M4",
            "Majority baseline",
            "Fold-10 macro-AUROC json from a train-frequency predictor. HYP is near chance.",
            "Labels or split are wrong. Do not start the CNN.",
          ],
          [
            "M5",
            "CNN on fold 10",
            "BCEWithLogits, early stop on fold 9. Saved curves plus fold-10 macro-AUROC and per-class AUROC. Macro-AUROC beats M4.",
            "Debug data or loss. Do not start M6.",
          ],
          [
            "M6",
            "Transformer on fold 10",
            "Same split, metrics, and seed protocol as M5. One architecture. Same artifact shape as M5.",
            "Skip. Demo and report use the CNN. Still a finished project.",
          ],
          [
            "M7",
            "Dropped-lead table",
            "One table: clean, V1–V3 zeroed, I+II only. Absolute macro-AUROC per model you actually trained.",
            "Fix evaluation. Do not invent SSL to explain a bad table.",
          ],
          [
            "M8",
            "Demo + report",
            "Runnable demo: pick a test ECG, mute leads, five scores update. Report has methods, the M7 table, and a not-a-diagnostic-system paragraph.",
            "The project is not done until both exist.",
          ],
        ]}
        rowTone={[
          "info",
          "info",
          "info",
          "info",
          "success",
          "warning",
          "success",
          "success",
        ]}
      />
      <Callout tone="warning" title="Stop rules">
        After M8, stop. Do not open SSL because M6 was easy. Polish the demo
        if you still want to type. If M6 is late, skip it and go to M7 with
        the CNN. An unfinished Transformer still ships. An unfinished
        pretraining run does not.
      </Callout>
      <Callout tone="info" title="Report language">
        Say you trained the models you actually finished and measured them
        under dropped leads. Do not say self-supervised pretraining improved
        robustness.
      </Callout>

      <H2>GPU budget</H2>
      <Text>
        Local machine: MacBook Air M3, 16 GB unified, 10-core GPU. Colab Pro
        is available. Use it for M5 and M6 on a T4. Do not select A100. These
        models use under 2 GB of VRAM. An A100 burns several times the compute
        units for no accuracy gain.
      </Text>
      <Row gap={24} wrap>
        <Stat value="T4" label="Colab GPU to request" tone="success" />
        <Stat value="~2 CU" label="M5+M6 on T4, order of magnitude" />
        <Stat value="A100" label="Do not request" tone="warning" />
        <Stat value="Air" label="M1–M4 and the demo" />
      </Row>
      <Table
        headers={["Item", "Size / cost", "Note"]}
        rows={[
          [
            "PTB-XL zip (both rates)",
            "1.7 GB down / 3.0 GB unzipped",
            "Skip records500. 100 Hz .dat files are ~0.5 GB.",
          ],
          [
            "All 100 Hz ECGs in RAM",
            "21,799 × 12 × 1000 × 4 B ≈ 1.05 GB",
            "Tight on 16 GB unified with Cursor and a browser. Prefer a memmap or on-the-fly WFDB reads.",
          ],
          [
            "Small 1D CNN",
            "≈ 0.5–5M params, < 1 GB train",
            "M5. Batch 32–64 on MPS.",
          ],
          [
            "Planned Transformer",
            "≈ 1–2M params, < 2 GB train",
            "300 tokens, dim 128, 4 layers. Attention is cheap at n=300.",
          ],
          [
            "One CNN run (M5)",
            "≈ 15–40 min on a T4; ≈ 0.5–3 h on this Air",
            "Estimate. 30–50 epochs, early stop on fold 9. Save a checkpoint every epoch.",
          ],
          [
            "One Transformer run (M6)",
            "≈ 20–60 min on a T4; ≈ 1–4 h on this Air",
            "Same data. T4 is the default. Skip M6 only if the run is buggy, not because of GPU.",
          ],
          [
            "Colab Pro compute",
            "T4 ≈ 1.2 CU/h. M5+M6 ≈ a couple of units",
            "Pro quota is ~100 CU/month. This project is noise on that quota if you stay on T4.",
          ],
          [
            "M7 dropped-lead eval",
            "minutes",
            "Three forward passes on ~2.2k test records. No training.",
          ],
          [
            "M8 demo",
            "CPU is enough",
            "One ECG, one checkpoint.",
          ],
          [
            "Parked SSL",
            "5–20 h class, not this cut",
            "That is the job that wants a cloud GPU. You are not doing it.",
          ],
        ]}
        rowTone={[
          "info",
          "warning",
          "success",
          "success",
          "info",
          "info",
          "success",
          "success",
          "success",
          "neutral",
        ]}
      />
      <Text size="small" tone="secondary">
        Wall-clock numbers are order-of-magnitude estimates for the coursework
        architectures on 100 Hz PTB-XL, not a benchmark of this laptop.
        Source: PhysioNet PTB-XL v1.0.3 file sizes; param/VRAM from the planned
        shapes; Colab T4 compute-unit rate ~1.2 CU/h (McCormick, Mar 2026).
        Times are estimates, not a benchmark of this laptop.
      </Text>
      <Callout tone="info" title="Split of labor">
        M1–M4 and packing a 100 Hz tensor live on the Air. Upload that tensor
        plus labels to Drive once. Colab runs train.py on a T4 against the
        packed file, not WFDB on Drive. Download checkpoints. M8 demo runs
        locally. The repo is the source of truth. The notebook is a launcher.
      </Callout>
      <Callout tone="warning" title="The 16 GB ceiling still matters">
        Unified memory is shared with the OS. Do not unzip records500 on the
        Air. Convert 100 Hz to one memmap or .pt once. If local MPS OOMs, drop
        batch size to 16 or move only the training step to Colab.
      </Callout>
      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>Do not start</CardHeader>
          <CardBody>
            <Text size="small">
              Masked-lead SSL. Time-patch MAE. Label-fraction sweeps. Three
              seeds. Bootstrap CIs. MIT-BIH noise. External datasets.
              Reliability diagrams. UMAP. Occlusion maps. Einthoven-aware
              reconstruction baselines.
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>After M8 only</CardHeader>
          <CardBody>
            <Text size="small">
              Gaussian noise at one SNR in the demo. Train-time lead dropout
              as a demo trick, not as a paper control. Gradio chrome. Never
              SSL.
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      <H2>Original review (why SSL was parked)</H2>
      <Callout tone="danger" title="If someone later wants a paper">
        Do not treat masked-lead MSE pretraining on PTB-XL as an open
        question. MaeFE, Oh et al., ST-MEM, and TolerantECG already ran
        versions of it. The holes below still apply. They are not homework
        for this course cut.
      </Callout>

      <H2>Three ways the conclusion stays uninterpretable</H2>
      <Grid columns={3} gap={12}>
        {KILL_SHOTS.map((shot) => (
          <div key={shot.id}>
            <Card>
              <CardHeader>{shot.title}</CardHeader>
              <CardBody>
                <Text size="small">{shot.body}</Text>
              </CardBody>
            </Card>
          </div>
        ))}
      </Grid>

      <H2>Prior art the plan does not cite</H2>
      <Text tone="secondary" size="small">
        Source: papers retrieved 12 Sep 2026. This is not an exhaustive survey.
        It is enough to retire the current research question.
      </Text>
      <Table
        headers={["Paper", "Citation", "Why it collides", "Link"]}
        rows={PRIOR_ART}
        rowTone={[
          "danger",
          "danger",
          "warning",
          "danger",
          "info",
          "info",
        ]}
      />

      <H2>What the hypothesis actually tests</H2>
      <Text>
        Written question: does masked-lead SSL improve robustness to missing
        leads and corruption versus supervised CNN and Transformer baselines?
      </Text>
      <Text>
        Actual test if executed as written: does a small Transformer that saw
        zeros in some channels during a reconstruction pretext lose less
        AUROC than a CNN that never saw missing leads, on a random mix of
        algebraically reconstructable and non-reconstructable masks, after
        both were trained on the same 17k labeled PTB-XL records?
      </Text>
      <Text tone="secondary">
        That is a different question. A yes is still compatible with “dropout
        augmentation would have done it” and with “the model memorized
        Einthoven.”
      </Text>

      <H2>Protocol holes</H2>
      <Row gap={8} wrap>
        {(["all", "blocker", "high", "medium"] as Filter[]).map((id) => (
          <span key={id}>
            <Pill active={filter === id} onClick={() => setFilter(id)}>
              {id === "all" ? "All" : severityLabel(id)}
            </Pill>
          </span>
        ))}
      </Row>
      <Table
        headers={["Severity", "Issue", "Why it bites", "Fix"]}
        rows={protocolRows.map((row) => [
          severityLabel(row.severity),
          row.issue,
          row.why,
          row.fix,
        ])}
        rowTone={protocolRows.map((row) => severityTone(row.severity))}
        striped
      />

      <Callout tone="warning" title="MVP success is not scientific success">
        CNN plus supervised Transformer under missing leads, with no SSL, is a
        lab report. Published CNNs already sit near 0.93 macro-AUROC on clean
        PTB-XL superdiagnosis. Missing-lead drops for standard 1D CNNs are also
        documented. That is the coursework cut. Do not write it up as
        answering the SSL question.
      </Callout>

      <H2>PTB-XL facts the plan underuses</H2>
      <Text size="small" tone="secondary">
        Superclass counts from PhysioNet PTB-XL v1.0.3. Sum exceeds 21799
        because labels co-occur.
      </Text>
      <BarChart
        horizontal
        categories={["NORM", "MI", "STTC", "CD", "HYP"]}
        series={[
          {
            name: "Records with label",
            data: [9514, 5469, 5235, 4898, 2649],
            tone: "neutral",
          },
        ]}
        height={200}
      />
      <Grid columns={2} gap={16}>
        <Stack gap={8}>
          <H3>Already in the metadata</H3>
          <Text>
            static_noise, burst_noise, baseline_drift, electrodes_problems are
            labeled. The plan synthesizes Gaussian noise and drift instead of
            slicing the real artifacts.
          </Text>
          <Text>
            Folds 9 and 10 are the high-quality human-validated folds. Using
            them as val/test is correct. Using fold 10 for model selection is
            not. The plan says this. Keep it.
          </Text>
        </Stack>
        <Stack gap={8}>
          <H3>Easy to get wrong</H3>
          <Text>
            Aggregate scp_codes through scp_statements.csv where diagnostic==1,
            then map diagnostic_class. Do not roll your own five-class scheme.
          </Text>
          <Text>
            21799 records, 18869 patients. Multiple ECGs per patient exist.
            strat_fold already respects that. The M2 overlap check is a
            test of your loader, not a new split.
          </Text>
        </Stack>
      </Grid>

      <H2>What to keep</H2>
      <Table
        headers={["Keep", "Why"]}
        rows={KEEP}
        columnAlign={["left", "left"]}
      />

      <CollapsibleSection title="Parked: research rewrite" count={1} defaultOpen={false}>
        <Stack gap={8}>
          <Text>
            On PTB-XL superdiagnosis, does masked-precordial reconstruction
            pretraining beat the same Transformer trained with identical
            lead-dropout, when test masks are clinically structured rather
            than random?
          </Text>
          <Text>
            That question needs the dropout control, precordial-only masks,
            bootstrap CIs, and a published CNN baseline. It is more than this
            course cut.
          </Text>
        </Stack>
      </CollapsibleSection>

      <Divider />

      <CollapsibleSection title="Repo and process nits" count={6} defaultOpen={false}>
        <Stack gap={8}>
          <Text>
            The proposed tree is fine. notebooks/ will rot. Put the audit in a
            script that writes figures/, not a notebook that is the pipeline.
          </Text>
          <Text>
            requirements.txt without pins is not a reproducible environment.
            Pin torch, wfdb, numpy, pandas, scikit-learn. Record the CUDA
            build. A lock file beats a README bullet.
          </Text>
          <Text>
            “Fixed random seeds” is not enough for CUDA. Set
            cudnn.deterministic, document remaining non-determinism, and run
            three seeds for the final table. One seed is a point estimate.
          </Text>
          <Text>
            A debugging dataset of 100 records must preserve label co-occurrence
            and at least two patients with multiple ECGs. Random 100 records
            can be all NORM.
          </Text>
          <Text>
            Do not store PTB-XL in git. data/ should be a download script plus
            a checksum against the PhysioNet v1.0.3 archive.
          </Text>
          <Text>
            An external set (Chapman-Shaoxing or CPSC2018) is a real
            generalization test. It is not in the coursework cut. It beats
            t-SNE if you ever add a milestone after M8.
          </Text>
        </Stack>
      </CollapsibleSection>

      <CollapsibleSection
        title="Architecture nits for the Transformer"
        count={4}
        defaultOpen={false}
      >
        <Stack gap={8}>
          <Text>
            300 tokens, dim 128, 4 heads, 3–4 layers is a reasonable first
            net. It is not a competitor to ECG-FM. Do not describe it as a
            foundation model.
          </Text>
          <Text>
            Pooling is unspecified. Mean pool over tokens will mix masked and
            live leads unless you mask the pool. CLS token or lead-masked mean
            pool. Pick one and test that it ignores padded leads.
          </Text>
          <Text>
            A reconstruction head that sees all 12 leads can cheat by copying
            correlated neighbors. ST-MEM’s lead-wise decoder exists for this
            reason. If you reconstruct, reconstruct per lead from that lead’s
            tokens plus a shared encoder, not from a full 300-token dump.
          </Text>
          <Text>
            Dropout 0.1 is fine. Layer-wise LR decay and a lower fine-tune LR
            matter more. MAE papers overfit PTB-XL-scale data without them.
          </Text>
        </Stack>
      </CollapsibleSection>

      <Text size="small" tone="tertiary" style={{ color: theme.text.tertiary }}>
        Coursework cut locked 12 Sep 2026. SSL remains parked. Repo is empty.
      </Text>
    </Stack>
  );
}
