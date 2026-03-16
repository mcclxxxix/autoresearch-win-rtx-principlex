Skip to content
mcclxxxix
autoresearch-principlex
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Owner avatar
autoresearch-principlex
Public
mcclxxxix/autoresearch-principlex
Go to file
t
Name		
mcclxxxix
mcclxxxix
Add files via upload
98d5331
 · 
11 minutes ago
LICENSE
Initial commit
8 hours ago
README.md
Update README by removing iframe and modifying links
6 hours ago
program-plx.md
Add files via upload
11 minutes ago
Repository files navigation
README
MIT license
autoresearch-principlex
Don't just find improvements. Know which ones will survive.

This repository is a fork of jsegov/autoresearch-win-rtx, which is itself a fork of karpathy/autoresearch. The purpose of this fork is to add a minimal interpretability constraint — one bit per commit — that filters out improvements which won't transfer across scales.

progress

The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. — @karpathy, March 2026.

The question Principlex asks: can we make comprehension scale alongside capability, before generation 10,205 arrives?

The problem
Autoresearch generates ~100 experiments overnight. Most are discarded. The keepers improve val_bpb but nobody knows why. The experiment log is two columns of DISCARDED entries, a handful of kept improvements, and no way to know why the kept ones worked except that the number went down. Improvements transfer from d12 to d24, but nobody can say which will transfer to d96, or to a different dataset, or to a different architecture, without re-running the whole search.

The bet
A single structural diagnostic — residual stream direction norms collapsed to a Gini coefficient — is enough to separate factored improvements from patchwork. Factored improvements transfer. Patchwork doesn't. This costs ~5% overhead per experiment.

What Principlex adds
One file: program-plx.md replaces program.md with a five-step experiment loop:

Hypothesis — write what you expect and why, before touching code.
Implement — edit train.py. One idea per experiment.
Run — 5-minute fixed time budget, same as upstream.
One-bit gate — render directional colors (residual stream L2 norms), compute Gini coefficient. Gini < 0.45 → factored → commit_bit = 1. Gini > 0.55 → patchwork → commit_bit = 0.
Decide — improved val_bpb AND bit = 1 → commit. Improved val_bpb AND bit = 0 → shelve. Worse val_bpb → revert.
Zeros get shelved, not discarded. They stay on branches for future investigation. But they don't pollute the main line.

The three-file simplicity is preserved. The agent still only touches train.py. The diagnostic snippet lives inside train.py at the end of training. No new Python files.

Fork scope
Upstream source: jsegov/autoresearch-win-rtx → karpathy/autoresearch
Primary objective: add a minimal interpretability gate to the autonomous research loop.
Scope of changes: program-plx.md (agent instructions), diagnostic snippet in train.py, cluster tooling.
Platform support: inherits jsegov's consumer GPU support (Turing >=8 GB, Ampere/Ada/Blackwell >=10 GB) plus SLURM cluster support for multi-agent runs.
All upstream compatibility and design choices remain intact.
How it works
Four files matter (three from upstream + one new):

prepare.py — fixed constants, one-time data prep, runtime utilities. Do not modify.
train.py — the single file the agent edits. Model, optimizer, training loop, and the Principlex diagnostic hook at the end. This file is edited and iterated on by the agent.
program-plx.md — Principlex agent instructions. The one-bit gate, the hypothesis-first loop, the shelving policy. This file is edited and iterated on by the human. Drop-in replacement for program.md.
plx-cluster.sh — SLURM script for running multiple agents in parallel on a GPU cluster. Optional. Not needed for single-GPU use.
The metric is still val_bpb (validation bits per byte) — lower is better. Principlex adds a second signal: commit_bit — 1 means factored, 0 means patchwork. Both are logged to results.tsv.

Quick start (single GPU)
Requirements: A single NVIDIA GPU (consumer desktop or datacenter), Python 3.10+, uv.

# 1. Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time)
uv run prepare.py

# 4. Run a single training experiment (~5 min)
uv run train.py

# 5. Verify the Principlex diagnostic fires
grep "principlex_gini" run.log
Quick start (Linux / cluster)
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup
uv sync
uv run prepare.py

# 3. Single agent
uv run train.py

# 4. Multi-agent on SLURM (e.g., 4 agents on 4 GPUs)
chmod +x plx-cluster.sh
./plx-cluster.sh --agents 4 --phase 2
Running the agent
Spin up Claude Code, Codex, or any coding agent in this repo, then prompt:

Read program-plx.md and let's kick off a new Principlex experiment. Do the setup first.
The agent will create a branch, initialize results tracking, and begin the hypothesis → implement → run → gate → decide loop.

Multi-agent cluster mode
plx-cluster.sh handles SLURM submission for parallel agents. Each agent gets its own git branch and working copy. Coordination is file-based: agents write results to a shared directory and read each other's logs to avoid duplicate experiments.

# Launch 8 agents for overnight run (Phase 2 = one-bit gate active)
./plx-cluster.sh --agents 8 --phase 2 --time 08:00:00

# Monitor
squeue -u $USER
tail -f ~/principlex-shared/logs/agent-01-*.out

# Aggregate results when done
bash ~/principlex-shared/plx-aggregate.sh ~/principlex-shared
Phases:

Phase 1 — Baseline. No gate. Runs original autoresearch loop for comparison data.
Phase 2 — Gated. One-bit gate active. Standard Principlex operation.
Phase 4 — Scale ladder. Agents at d12/d24/d48/d96 with automatic promotion of surviving changes across scales.
The one-bit gate, explained
Why one bit? Because factored structure and comprehensibility aren't independent properties — they're the same thing measured at different levels. That's the core insight from PicBreeder research: NEAT-evolved CPPNs develop organized, inspectable internal circuitry while SGD-trained networks producing the same outputs develop disorganized patchwork. Same capability, radically different comprehensibility. The organization is the legibility.

Two axes collapse to one bit. Factored or not. Transfers or doesn't.

The diagnostic: hook the residual stream at each transformer block, capture L2 norms per position, compute the Gini coefficient across all layers. A factored model distributes activation mass evenly (low Gini). A patchwork model concentrates it in hot clusters (high Gini). Low Gini correlates with cross-scale transfer. High Gini correlates with improvements that vanish at larger scales.

Cost
Baseline autoresearch	Full Principlex (4 roots)	One-Bit Principlex
Overhead	0%	10–20%	~5%
Descriptors	None	Engram + Cuneiform + Dir. Colors + Tab2Visual	1 bit + optional engram
Transfer signal	None (re-run everything)	High (expensive)	High (cheap)
On a consumer GPU at 12 experiments/hour, the one-bit gate costs you roughly 1 experiment slot per cycle. You drop from ~12 to ~10–11 experiments/hour. That's the concrete cost of knowing which improvements will survive.

Project structure
prepare.py        — constants, data prep + runtime utilities (do not modify)
train.py          — model, optimizer, training loop + diagnostic hook (agent modifies this)
program-plx.md    — Principlex agent instructions (human modifies this)
program.md        — original baseline instructions (kept for reference)
plx-cluster.sh    — SLURM multi-agent launcher (optional)
AGENDA.md         — 10-week research plan for systematic validation
principlex.html   — interactive demo of the network and strategy
pyproject.toml    — dependencies
Design choices
Everything from upstream applies, plus:

One bit, not four roots. The original Principlex proposal (#226 on autoresearch) had four visualization requirements: Engrams, Cuneiform, Directional Colors, Tab2Visual. That's too much descriptor for a binary signal. This fork uses one: directional colors collapsed to a scalar Gini coefficient. One bit per commit.
Shelved ≠ deleted. Patchwork improvements are real at the current scale. They're kept on branches and logged. They just don't merge to main. If the gate is later found to be too strict, shelved experiments can be recovered.
File-based coordination. Multi-agent mode uses no databases, no coordination servers, no APIs. Just a shared filesystem and git. This runs anywhere SLURM runs.
Diagnostic inside train.py. The residual stream hook is a few lines of Python appended to the end of training. No separate analysis scripts. Three-file simplicity is preserved.
Platform support
Inherits the full platform matrix from jsegov/autoresearch-win-rtx:

Architecture	Min VRAM	Example GPUs
Turing	>=8 GB	RTX 2060 12GB, RTX 2070, RTX 2080 Ti
Ampere	>=10 GB	RTX 3060 12GB, RTX 3080, RTX 3090
Ada	>=10 GB	RTX 4070, RTX 4080, RTX 4090
Blackwell	>=10 GB	RTX 5070, RTX 5080, RTX 5090
Datacenter	any	H100, H200, A100 (via SLURM cluster mode)
Runtime path: PyTorch SDPA attention + eager optimizer steps. No FA3, no torch.compile.

Related work
karpathy/autoresearch — the original
jsegov/autoresearch-win-rtx — Windows/consumer GPU fork (direct upstream)
miolini/autoresearch-macos — macOS fork
trevin-creator/autoresearch-mlx — Apple Silicon MLX fork
mutable-state-inc/autoresearch-at-home — SETI@home-style collaborative fork
Principlex #226 on karpathy/autoresearch — the original proposal by McClurkin https://specialtyconsultants.co/autoresearch/
License
MIT. Credits autoresearch (karpathy) and autoresearch-win-rtx (jsegov) as upstream.

About
Zurada experiment

Resources
 Readme
License
 MIT license

