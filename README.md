# Critical Analysis of FinAgent (KDD 2024) — Cross-Domain Probe on LOB

A short, reproducible study examining whether the **agentic LLM architecture**
proposed in FinAgent (Zhang et al., KDD 2024) is genuinely the source of its
trading skill, or whether the same skill is recoverable through a numerical
analog of its *regime-conditional* decision rule — at four orders of
magnitude lower latency.

> **Submitted to:** «Лето с AIRI 2026» summer school
> **Track:** AgenticAI — *autonomous agents: self-evolving systems,
> researcher-agents and industrial solutions*
> **Author:** Sergei Solovev, MSc candidate in AI, HSE FCS × HSE NN / Neimark
> Institute · [sesesolovev@edu.hse.ru](mailto:sesesolovev@edu.hse.ru)
> **Date:** May 2026

---

## TL;DR

| Setting | Pearson ρ on LOB mid-price | Latency / decision |
|---|:---:|:---:|
| Regime-blind Ridge (baseline) | +0.2592 | ≈ 1.3 ms |
| **Ridge + 1 regime feature** (this work) | **+0.2656** | ≈ 1.3 ms |
| DA-BiGRU-CNN (own preprint, figshare 31859557) | +0.266 | ≈ 1.2 ms |
| FinAgent (LLM-based, KDD 2024) | reported in paper | **≈ 2 s** (× 20 000 over HFT budget) |

**Finding.** On the limit-order-book data, a simple Ridge regression
augmented with one volatility-regime indicator matches the skill of a deep
domain-aware network *and* approximates the qualitative gain FinAgent claims
from its LLM-based reflection module — at five orders of magnitude lower
latency than the LLM-in-the-loop. Almost all the regime-conditioning gain
appears in the calm regime (Δρ = +0.015); in the volatile regime it
vanishes.

This is **not** a refutation of agentic LLMs. It is a falsification of the
specific claim that the LLM is the necessary substrate for the
reflection-driven gain reported in FinAgent — at least in the HFT-LOB
regime.

---

## Paper under analysis

> Zhang Y., Yuan Y., Chen S., Sun J., Hou Y., Wang Z., Zhang X., An B.
> *FinAgent: A Multimodal Foundation Agent for Financial Trading:
> Tool-Augmented, Diversified, and Generalist.*
> **KDD 2024**, Research Track, Core A\*.
> arXiv: <https://arxiv.org/abs/2402.18485>
> Original code: <https://github.com/Mookiee/FinAgent>

The full critical analysis (method, strengths/weaknesses, proposed
extensions) is in [`research_proposal.pdf`](research_proposal.pdf) — a
2-page document submitted to AIRI as the Research Proposal artifact.

## Repository contents

| File | Description |
|---|---|
| [`research_proposal.pdf`](research_proposal.pdf) | 2-page critical analysis (RU), AIRI Research-Proposal artifact |
| [`presentation.pdf`](presentation.pdf) | 16-slide companion deck (RU), same visual identity |
| [`report.ipynb`](report.ipynb) | Reproducible Jupyter notebook with executed cells, embedded outputs, and pedagogical commentary |
| [`experiment.py`](experiment.py) | Standalone Python script — runs the full experiment end-to-end in ≈ 60 s on CPU |
| [`metrics.json`](metrics.json) | Raw metrics from the run (Pearson ρ, MAE, per-regime, latency) |

## How to reproduce

```bash
# 1. Dependencies (CPU only, no GPU/LLM API required)
pip install numpy pandas scikit-learn scipy pyarrow

# 2. Data
# The LOB dataset (Wunder Fund RNN challenge) is local-only —
# this is the same dataset used in the author's LOB preprint
# (figshare 31859557). The repository expects:
#   <project_root>/datasets/valid.parquet
# If you have access to a similar LOB dataset, point DATA in
# experiment.py at it.

# 3. Run
python experiment.py
# End-to-end on a recent laptop: ~60 seconds.
```

The notebook `report.ipynb` is **pre-executed** — outputs are embedded, so a
reviewer can read it without re-running the kernel.

## Method (one paragraph)

We construct a cheap numerical proxy for FinAgent's text-based reflection
module: a rolling standard deviation of the mid-price proxy `(p₀+p₁)/2` over
a 100-tick window, median-split into *calm* / *volatile* states. We compare
two predictors of the short-horizon mid-price return: **(A)** a
regime-blind Ridge regression on 32 LOB features and **(B)** a two-specialist
Ridge mixture where each specialist is trained on one regime. Both train on
the first 70 % of a sequential 200 000-row sample and test on the remaining
30 %, with no leakage across `seq_ix`.

## Key findings

1. **Regime-conditioning helps, but only in the calm regime.** Δρ = +0.015
   on calm test rows; Δρ ≈ 0 on volatile test rows. This structurally
   explains why FinAgent transfers well to daily-frequency DJ30 trading
   but cannot transfer to HFT.
2. **A simple Ridge + one regime feature matches a deep network.** Ridge
   mixture ρ = 0.2656 ≈ DA-BiGRU-CNN ρ\_w = 0.266 from the author's LOB
   preprint. Direct empirical illustration of the *feature sufficiency*
   thesis from that paper.
3. **Latency gap of × 20 000** between FinAgent's LLM-based decision step
   and a typical HFT market-maker budget. This is a structural gap, not an
   engineering one: distilling FinAgent's modules into a small network
   removes precisely the reflection and tool-use blocks that motivate the
   paper.

## Related work by the author (figshare)

All on the [author's figshare profile](https://figshare.com/authors/Sergei_Solovev/23264342):

- **When Less Is More**: Domain-Aware Dual-Branch Recurrent Networks for
  LOB Mid-Price Prediction (figshare 31859557) — the deep model whose skill
  this work reproduces with a much simpler architecture
- **When Retrieval Hurts**: An Honest Evaluation of RAG for Solidity
  Vulnerability Detection (figshare 32141182) — the methodological prior
  showing a popular technique (RAG) does not help at larger sample sizes
- **AI-Managed ERC-4626 Yield Vault** with Multi-Criteria Decision Making
  (figshare 32141167) — a complementary positive result: *where* agentic AI
  in DeFi does demonstrably work, with formal verification
- Machine Learning-Based Vulnerability Detection in Ethereum Smart
  Contracts (figshare 31429971)
- OCR-Based vs End-to-End Transformer Pipelines for Receipt Information
  Extraction (figshare 31430086)

## Methodological signature

The line of work this repository sits in follows a consistent pattern:
*if a popular technique does not give measurable skill, show it honestly
and explain structurally why.* Three companion negative findings:

- LOB: deep-ensemble averaging hurts (negative ensemble effect)
- RAG: naive retrieval hurts at n ≥ 250 (sample-size reversal)
- Agents (this work): LLM-in-the-loop is structurally over budget for HFT-LOB

This is **not** «agents don't work». It is «here is the structural
boundary where they don't, and why».

## Citation

If you use this analysis or its numerical results, please cite the AIRI
Research Proposal:

```bibtex
@misc{solovev2026finagentprobe,
  author       = {Solovev, Sergei},
  title        = {{Critical Analysis of FinAgent (KDD 2024) and Cross-Domain
                   Probe on Limit-Order-Book}},
  year         = {2026},
  howpublished = {Research proposal submitted to «Лето с AIRI 2026», AgenticAI track},
  url          = {https://github.com/SergeySolovyev/airi-2026-finagent-probe}
}
```

## License

[MIT](LICENSE). You may reuse, modify, and redistribute the code freely;
the academic findings should be cited as above.
