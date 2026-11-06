# AGENTS.md

Operating contract for agents in the tiny-dit repo.

## What this repo is

A from-scratch toy text-to-image DiT on one consumer GPU. The design is agreed
(`docs/SPEC.md`); `docs/HANDOVER.md` says what is built and measured. `README.md`
has the stages and layout.

## Documentation map

- `README.md` — what the project is, layout, setup.
- `docs/SPEC.md` — the settled design, in present tense. The source of truth.
  **Carries no history**: when a rule changes, SPEC states the new rule, nothing
  about what it replaced. Every change to it is a design change and goes to Daniel.
- `docs/PLAN.md` — the open-questions ledger, grouped: what is unsure, how each
  would be settled, current evidence. Questions only; measured numbers live in FACTS. Read
  before running an experiment or changing a default.
- `docs/FACTS.md` — a dense index of insights and the measured numbers behind them, each with source
  and date; replaced, not appended to, when a new read supersedes it. Not a logbook.
- `docs/log/` — the logbook: one note per run or read, `YYYY-MM-DD-title.md`, with the full tables;
  FACTS and PLAN point to them where helpful.
- `docs/archive/` — closed questions and decisions, one note per session or
  closure, named `YYYY-MM-DD-kebab-title.md`. Read for the reasons; never rewrite.
- `docs/topics/` — one doc per line of work Daniel opens as its own topic (its plan, design and reads), linked
  from its PLAN question; numbers still go to FACTS, closures to `archive/`.
- `docs/HANDOVER.md` — session-to-session orientation: where things stand, next
  steps, traps.
- `docs/REFERENCES.md`, `docs/references/` — sources and what each establishes;
  verbatim source material.

## How we work from PLAN

When a question closes, Daniel decides it. Its conclusion goes into a new note in
`docs/archive/` with the reason, and the question leaves PLAN in the same session.
If the answer settles design, SPEC gets the result and only the result. A new
question goes into PLAN when it is found. A measurement goes into FACTS, and
moves a PLAN answer when it bears on one. When a finding contradicts something
already written, correct it where the wrong claim lives. Nothing is settled
because it is written down; measurement beats the docs.

Work one PLAN group at a time: the questions in a group share runs or code.

## Rules

- `docs/SPEC.md` is the source of truth. Implement what it says. Any deviation,
  however small, is a design change: stop and bring it to Daniel with the facts
  that decide it. Do not "improve" the design silently.
- An observation is not a fix request. Report it; change nothing until asked.
- Fail early: verify the risky assumption before building on it, with a small
  script, and report the numbers. The listed ones are in PLAN (groups A and B):
  Primal latent stats, EmbeddingGemma 2 output norm, Usui's handling of 1-D
  params.
- Never discard a working state without isolating the cause of a problem.
- Pure PyTorch. No HF Trainer, no Accelerate, no ai-toolkit.
- Logs are JSONL in the run directory, one object per line, so they can be read
  directly. wandb mirrors them for live curves (SPEC §8); the JSONL is the record.
- EmbeddingGemma 2 runs in bf16 or fp32, never fp16.
- Reports say what was verified, what was assumed, and what is still open.

## Commits

Agents own commits in this repo. Daniel takes part in design, not code review:
a subagent reviews each unit against SPEC and these rules, the session judges the
review and commits. One commit per coherent change: a design change, a probe and
a doc correction are separate commits. Subject says what changed; a body when
intent or impact needs it. Pushing stays with Daniel.

## Running things

From the repo root, `uv sync`. torch 2.15 comes from the PyTorch cu132 index.
usuitrack is an editable path dependency from `../usuitrack-zero-init`, the worktree
of usuitrack branch `zero-init` (archive, 2026-11-01), so it runs whatever is checked
out there.

A run streams to wandb (project `wandb_project` in its config) with the entity and key
from `$WANDB_ENTITY` and `$WANDB_API_KEY`; `WANDB_MODE=offline` runs without the network.
A run whose `wandb.init` fails leaves its run directory behind: delete it before rerunning.

Datasets, latents, embeddings and runs live under `/mnt/mars/ai/tiny-dit/`, not in
the repo: `/home` has too little space for ~160 GB of latents.

The GPU (RTX 4070 Super, 12 GB) is shared with ai-toolkit jobs. Check
`nvidia-smi` before launching. **Launch training jobs in the background, never in
the foreground.**

## Code style

- Every line earns its place against the task. No flags, branches, adapters,
  warnings, or "just in case" scaffolding. When the shape is wrong, prefer
  deletion or a sharper boundary over accommodation.
- No parallel compiled and uncompiled implementations of the same maths; they
  drift silently. A compiled path is the only implementation, or the eager path
  calls into the same function.
- Boring and clear over clever; explicit domain concepts over generic plumbing.
- Direct, readable PyTorch. Comments for non-obvious math, tensor-shape
  invariants, and performance traps, never to narrate an obvious operation.
- Match the surrounding code's density, naming, and idiom.

## Validation

```bash
uv run python -m pytest tests/
```

Do not report work as done because imports resolve or a smoke run descends. If
validation could not run, say exactly what is unverified.
