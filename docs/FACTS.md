# Facts

What holds in this project and has to be in mind before designing, running or reading anything: principles and
gotchas, each a statement that stays true until the code or the data changes. No measurements and no history; each
entry points to where its evidence lives. The measurements that stood here before 2026-11-20 are in
`log/2026-11-20-measurements-from-facts.md`.

## Reading a result

- The averaged loss is flat over long stretches and says little. Read the loss and R² by t, and R² across steps at
  one t, never across t: it falls with t by construction (SPEC §8).
- One seed is one draw. Before reading a difference, say what the seed spread at that length and size is; only one
  pair is measured (log 2026-11-18-seed42). A same-seed rerun of the same config reproduces its score, so two
  same-seed arms differ by the change and by the trajectory it sets off.
- Loss does not measure prompt following: the caption-against-null gap is small at every width. Prompt following is
  judged on samples and, partly, by the caption-use probe (`scripts/probe_caption_use.py`, a manual tool, run on a
  checkpoint when a conditioning change is read).
- The held-out set's captions are longer than Lunara's; a change to how the caption is read can show on one and not
  the other (log 2026-11-20-joint-attention-and-caption-refiner). Lunara is scored at 256 px, off the training
  resolution, so it reads how a model carries over to a larger grid.
- Throughput compares only at the same power limit and on a GPU shared with nothing; the limit runs are made at is in
  HANDOVER. Compare arms at the same wall-clock as well as the same step when their speeds differ.
- Short runs measure cost, not stability: the spikes of a run without the norm ceiling began late and grew with the
  run (log 2026-11-20-measurements-from-facts, Spikes).
- A constant-lr run understates where an anneal takes it; a cooldown is its own short run (SPEC §6). The late gain of
  a run follows its relative step.

## Model and optimizer

- The function is tied to the weight norms. Without a ceiling the norms grow without end and the run spikes; inside
  the ball of their init norm the matrices train without spikes (archive 2026-11-12-norm-ceiling).
- The slice is anchored and the meta channels are not. The slice is read out raw, with no final norm or projection,
  so its scale is the prediction's. The meta channels are only read through RMS norms, blind to their scale, and
  nothing in the loss holds their size: it grows with depth, partly as a per-image common mode, written at attention
  and some conv blocks (log 2026-11-21-stream-scale).
- The adaLN gates are large on the meta channels and small on the slice; the size is in the per-block tables. The
  branch norms cap a write before its gate, not the gate, and the tables are outside the norm ceiling, so nothing
  bounds the gates. A bound on them is not the identity (logs 2026-11-21-gate-sizes, 2026-11-21-stream-scale).
- Every branch normalizes its read over all channels, so as the meta part grows the slice, x_t and the prediction so
  far, fades from what deep blocks read. A probe of the stream reads slice and meta apart (`scripts/probe_stream.py`).
- Behind the zero gates every block starts as the identity and every upstream gradient is zero: on the first step
  of the Usui matrices only `ada.1.weight` moves, the patch embed and the blocks follow as the gates open, and a frame fitted to the first
  gradient has nothing to fit (SPEC §5, §6).
- On CUDA, Usui moves a matrix whose gradient is zero by a small relative amount on its first step; on the CPU the move
  is exactly zero. A test of "nothing moved" holds to a tolerance on CUDA.
- Usui's `grad_capture` under Molegrad is a probed estimate and can read above 1: comparable across runs and steps,
  not the true share.
- `torch.linalg.eigh` reads back to the host on CUDA, so each Usui step syncs and cannot be captured as one CUDA graph.
  At small batch the step is launch-bound, and the Usui step's fixed cost is most of it (PLAN H1).
- Blocks compiled while a Molegrad frame waits for its fit break their graphs at its zero-gradient branch and hit the
  recompile limit; `train.py` compiles once no frame waits. Compiled bf16 gradients are closer to fp32 than eager.
- flex_attention has no backward on the CPU and refuses parameters that require grad even in a forward: tests that
  take gradients run on CUDA (`tests/conftest.py`). A mask built inside the compiled graph fails to lower under
  no_grad; the mask is built by the caller (`model.joint_attention`).
- A checkpoint loads only into the model it was trained with.

## Data and encoders

- Primal's encoder is not equivariant to flips or rotations; its decoder is. A flipped latent decodes to the flipped
  image but is not what the encoder gives for it (PLAN B4).
- EmbeddingGemma 2 runs in bf16 or fp32, never fp16. On the CPU fp32 is the faster. Captions are encoded with no
  prompt prefix (archive 2026-11-12-no-prefix). Its text-only load needs torchvision installed and the vision and
  audio configs switched off (`scripts/precompute.py`).
- The same caption can give embeddings that differ by bf16 rounding; repeats are found by the latent's bytes, not
  the embedding's.
- Captions have rare outliers of thousands of tokens and defects (escaped markup, repeats); they are cut at 512 token
  states and the defective rows excluded (SPEC §1). An outlier in a micro-batch pads every row to its length.
- TF32 convolutions in the precompute change a latent by less than rounding it to bf16 does.
- A copy from pageable CPU memory into a CUDA tensor blocks the host until the queued GPU work has run; a pinned
  source copied `non_blocking` does not.
- pyarrow's int32 offsets cap a binary column chunk at 2 GB: take slices, or cast to `large_binary`.

## Machine and tooling

- The GPU (RTX 4070 Super, 12 GB) is shared. Without a desktop login the LUKS drive with the training set is not
  mounted, and the machine can suspend while a run trains: launch under `systemd-inhibit --what=sleep:idle`.
- `nvidia-smi -pl` needs `pkexec`, which opens a dialog on the desktop. After a driver update `nvidia-smi` fails with a
  driver/library mismatch until the reboot.
- `pgrep -f <name>` matches the shell wrapper of any agent command containing the name; find and kill runs by PID.
- wandb groups series on the first slash only: keep keys one level deep. A run's console ends with `shutdown can only
  be called once` from wandb; it is not a fault.
