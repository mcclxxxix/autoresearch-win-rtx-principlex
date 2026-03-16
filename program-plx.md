# program-plx.md — Agent Principlex

> One bit per commit. Factored structure transfers. Patchwork gets shelved.

This file replaces `program.md`. It is the agent instruction set for the Principlex
fork of autoresearch. The agent reads this file, follows the loop, and produces
interpretable experiments — not just lower numbers.

Credits: autoresearch (karpathy), autoresearch-win-rtx (jsegov), Principlex (#226).

---

## Setup (once per session)

1. **Create the branch:** `git checkout -b principlex/<tag>` from current master.
2. **Read the in-scope files:**
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
3. **Verify data exists:** Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
4. **Initialize results.tsv:** Create `results.tsv` with this header row:
   ```
   experiment	tag	val_bpb	peak_vram_mb	commit_bit	gini	notes
   ```
   The baseline will be recorded after the first run.
5. **Create viz/ directory:** `mkdir -p viz/` — this is where diagnostic renders go.
6. **Confirm and go.**

---

## The Experiment Loop

Each experiment follows this sequence. No exceptions. No shortcuts.

### Step 1 — Hypothesis

Before touching any code, write a one-line hypothesis in your reasoning. What do you
expect to change and why? Example: "Replacing learned positional embeddings with RoPE
should reduce parameter count without hurting val_bpb because rotary embeddings encode
relative position without stored weights."

### Step 2 — Implement

Edit `train.py` with your experimental change. Keep diffs minimal and reviewable.
One idea per experiment. Do not stack changes.

### Step 3 — Run at primary scale

```bash
uv run train.py > run.log 2>&1
```

Redirect everything. Do NOT use tee or let output flood your context.

Read out results:
```bash
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

If grep is empty, the run crashed. Run `tail -n 50 run.log` for the traceback.
If you can't fix it in 2 attempts, revert and move on.

### Step 4 — The One-Bit Gate

**This is what makes Principlex different from baseline autoresearch.**

If val_bpb improved (lower), you do NOT immediately commit. Instead:

1. **Render directional colors.** Add this diagnostic snippet to the end of your
   training run (or run it as a post-hoc script). It captures the residual stream
   direction at the final validation step and writes a simple heatmap:

   ```python
   # --- Principlex diagnostic: directional color render ---
   # Insert after final validation, before exit.
   # Captures residual stream L2 norms per position per layer.
   import json, os
   diag = {}
   model.eval()
   with torch.no_grad():
       x_val = next(iter(val_loader))[0][:1].to(device)  # single example
       # Hook residual streams
       norms_by_layer = []
       def hook_fn(module, input, output):
           if isinstance(output, tuple):
               output = output[0]
           norms = output.float().norm(dim=-1).squeeze(0).cpu().tolist()
           norms_by_layer.append(norms)
       hooks = []
       for block in model.transformer.h:
           hooks.append(block.register_forward_hook(hook_fn))
       model(x_val)
       for h in hooks:
           h.remove()
       diag['residual_norms'] = norms_by_layer
       diag['n_layers'] = len(norms_by_layer)
       diag['seq_len'] = len(norms_by_layer[0]) if norms_by_layer else 0
   os.makedirs('viz', exist_ok=True)
   with open('viz/directional.json', 'w') as f:
       json.dump(diag, f)
   print("principlex_diag: directional.json written")
   ```

2. **Check scale consistency.** If the cluster supports it (see multi-agent mode),
   re-run the same change at a larger scale (e.g., double `n_embd` or `n_layer`).
   Compare `viz/directional.json` between scales. You're looking for:
   - **Consistent pattern** → norms distribute similarly across layers/positions.
   - **Chaotic reorganization** → norm distribution shifts dramatically.

   On a single consumer GPU where you can't afford the second run, use a heuristic:
   compute the Gini coefficient of the residual norms across layers. Low Gini (<0.45)
   suggests factored structure. High Gini (>0.55) suggests patchwork.

   ```python
   import numpy as np
   all_norms = [n for layer in norms_by_layer for n in layer]
   sorted_norms = np.sort(all_norms)
   n = len(sorted_norms)
   index = np.arange(1, n + 1)
   gini = (2 * np.sum(index * sorted_norms) / (n * np.sum(sorted_norms))) - (n + 1) / n
   print(f"principlex_gini: {gini:.4f}")
   ```

3. **Assign the bit.**
   - Gini < 0.45 (or pattern holds across scale) → `commit_bit = 1`
   - Gini > 0.55 (or pattern scrambles) → `commit_bit = 0`
   - Gini 0.45–0.55 → borderline. Commit but flag as `BORDERLINE` in notes.

### Step 5 — Record and decide

Append to `results.tsv`:
```
<experiment_num>	<tag>	<val_bpb>	<peak_vram_mb>	<commit_bit>	<gini>	<notes>
```

**If val_bpb improved AND commit_bit = 1:** Keep the commit. This is a factored
improvement that is likely to transfer across scales.

**If val_bpb improved AND commit_bit = 0:** `git reset` back. The improvement is
real at this scale but structurally patchwork. Log it as `SHELVED` in notes. Do not
merge to the branch. These are candidates for future investigation, not current
integration.

**If val_bpb is equal or worse:** `git reset` back regardless of commit_bit.

### Step 6 — Repeat

Go to Step 1. You are a completely autonomous researcher. Keep iterating.

---

## Research Priorities

The agent should explore ideas in roughly this order, but exercise judgment:

1. **Architecture** — attention variants, positional encodings, activation functions,
   normalization strategies, width/depth tradeoffs.
2. **Optimization** — learning rate schedules, optimizer changes, gradient clipping,
   weight decay tuning, batch size experiments.
3. **Regularization** — dropout, weight tying, embedding sharing, data augmentation
   at the token level.
4. **Efficiency** — reducing parameter count while maintaining or improving val_bpb,
   memory optimization, faster convergence.

When proposing experiments, prefer changes that have a clear mechanistic hypothesis.
"I'm trying this because paper X showed it works" is weaker than "I'm trying this
because the current attention pattern wastes capacity on distant tokens that
TinyStories rarely needs."

---

## Rules

- The agent ONLY modifies `train.py`. Everything else is read-only.
- The diagnostic snippet goes inside `train.py` at the end of training, not in a
  separate file. Three-file simplicity is preserved.
- `viz/` directory outputs are untracked by git. They are ephemeral diagnostics.
- `results.tsv` is untracked by git.
- Do NOT commit changes that crash. Do NOT commit changes with commit_bit = 0.
- If you hit 3 consecutive crashes, step back and try a simpler experiment.
- If you hit 3 consecutive commit_bit = 0 results, reflect on why your changes
  are producing patchwork structure and adjust your research direction.

---

## Multi-Agent Mode (Cluster)

When running multiple agents on a SLURM cluster (see `plx-cluster.sh`), each agent
operates on its own branch: `principlex/<agent_id>`. Agents do NOT share branches.

Coordination is file-based:
- Each agent writes its `results.tsv` to a shared directory:
  `$SHARED_DIR/results/<agent_id>.tsv`
- Before starting a new experiment, the agent reads all `.tsv` files in
  `$SHARED_DIR/results/` to see what has been tried. Avoid duplicate experiments.
- The best commit_bit=1 result across all agents is the current global best.
  An agent can `git cherry-pick` a winning commit from another agent's branch.

No databases. No coordination servers. Just files and git.

---

## License

MIT. Credits autoresearch (karpathy) and autoresearch-win-rtx (jsegov) as upstream.
