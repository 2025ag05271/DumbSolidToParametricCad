# solid2param — Dumb solid → parametric design history

Goal: given a "dumb" B-Rep solid (STEP, no history), predict **a** valid parametric
feature history (JSON) that rebuilds it. Any history that reproduces the geometry is
acceptable — it does not need to match the original designer's history.

```
STEP → UV-Net B-Rep graph → encoder → Transformer decoder → token sequence → JSON → rebuild (CadQuery / CATIA)
```

New to the underlying theory (neural networks, GNNs, Transformers, B-Rep, NURBS, UV space,
continuity, kernel internals)? See the study guide: [THEORY.md](THEORY.md).

## Roadmap

| Step | Status | What |
|------|--------|------|
| 1 | ✅ | Operation vocabulary + JSON schema + reference rebuilder (`cadseq/`) |
| 2 | ✅ | Procedural generator → 20K (STEP, JSON, graph) triplets with validity checks (`datagen/`) |
| 3 | ⏳ | JSON → CATIA rebuild (route TBD: COM automation / CAA) |
| 4 | ✅ | Tokenizer (quantized, lossless JSON ↔ tokens) + dataset loader (`cadseq/tokenizer.py`, `prepare.py`, `dataset.py`) |
| 5 | ✅ | Encoder–decoder model, grammar-constrained decoding, training & CAD-level evaluation (`s2pnet/`) |
| 6 | ⏳ | Rebuild-verify-rerank at inference, then geometric reward fine-tuning |

## Layout

```
├── step_to_graph.py  # STEP → UV-Net face-adjacency graph (script version of the exploration notebook);
│                     #   the single implementation of UV-grid sampling used by the whole pipeline
├── cadseq/
│   ├── schema.py     # pydantic models = the JSON format (v1.0)
│   ├── build.py      # JSON → CadQuery solid → STEP (Builder applies features incrementally)
│   ├── graph.py      # pipeline access to step_to_graph (numpy arrays) + to_dgl()
│   ├── tokenizer.py  # JSON ↔ token ids (quantized 0.05 mm, canonical)
│   ├── grammar.py    # incremental token grammar → allowed-next-token sets (constrained decoding)
│   ├── prepare.py    # pack a dataset: tokens.npz, graphs/*.npy, splits.json, vocab.json
│   ├── dataset.py    # torch Dataset + collate (pure torch, memory-mapped)
│   └── metrics.py    # Monte Carlo IoU + Chamfer distance between solids
├── s2pnet/           # model (encoder.py, decoder.py, model.py), train.py, evaluate.py, predict.py
├── datagen/          # synthetic dataset generator (see below)
├── examples/         # SimpleRectangle.stp, simple_rectangle / bracket / flange JSON histories
├── THEORY.md         # study guide for the underlying theory
├── requirements-train.txt / requirements-datagen.txt
├── runs/             # (git-ignored) training runs: checkpoints, logs, eval results
└── data/             # (git-ignored) generated datasets (synth_v2/ = current)
```

`data/` and `runs/` are not in the repository (several GB; checkpoints exceed GitHub's 100 MB
file limit). Regenerate them with the commands below.

## STEP → graph (`step_to_graph.py`)

Script version of the `STEP_to_UVNet_Graph.ipynb` exploration notebook, with the same functions:
`load_step`, `tessellate_shape`, `render_shape`, `count_topo`, `check_brep`,
`sample_face_uv_grid`, `sample_edge_uv_grid`, `brep_to_graph_arrays`, `brep_to_dgl_graph`,
`graph_checks` and `plot_graph`. `cadseq.graph` re-exports the sampling functions, so dataset
generation, training data and prediction all use exactly this code.

```powershell
# topology, validity, graph sanity checks (+ optional plots / arrays / DGLGraph)
python step_to_graph.py examples\SimpleRectangle.stp --plots out\ --save-npz out\ --dgl
python step_to_graph.py path\to\folder_of_step_files
```

```python
from step_to_graph import load_step, brep_to_dgl_graph   # DGL optional; brep_to_graph_arrays needs only numpy
shape = load_step("examples/SimpleRectangle.stp")
graph, centroids = brep_to_dgl_graph(shape, N=10)        # ndata["x"] (6, 700), edata["x"] (24, 60)
```

## Usage

Two virtual environments (paths below assume both live in this folder):

| venv | Used for | Requirements |
|------|----------|--------------|
| `myenv` | dataset generation (`datagen`), DGL graphs | `requirements-datagen.txt` — torch 2.3 + DGL 2.2, numpy 1.x |
| `.venv` | training / evaluation / prediction (`s2pnet`), `cadseq.prepare` | `requirements-train.txt` — torch ≥ 2.7 + CUDA 12.8, required for Blackwell GPUs (sm_120); DGL has no Windows builds for this torch, so the model is pure PyTorch |

```powershell
# training venv (one-off)
py -3.11 -m venv .venv
$th = "--trusted-host","pypi.org","--trusted-host","files.pythonhosted.org","--trusted-host","download.pytorch.org"
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128 @th
.venv\Scripts\python.exe -m pip install -r requirements-train.txt @th
```

Run everything from the repository root:

```powershell
.venv\Scripts\python.exe -m cadseq.build examples\bracket.json --step out\bracket.step
```

In the datagen environment CadQuery must be installed without letting it upgrade numpy
(torch 2.3 / DGL 2.2 need numpy 1.x):

```powershell
$th = "--trusted-host","pypi.org","--trusted-host","files.pythonhosted.org"
pip install cadquery==2.8.0 --no-deps @th
pip install -r requirements-datagen.txt @th
```

`pip` will warn that cadquery wants `nlopt>=2.9`; that is expected (nlopt is only
used by CadQuery's assembly/sketch constraint solver, which we do not use).

## JSON format (v1.0)

Units are millimetres and degrees. A file is `{"version": "1.0", "units": "mm", "features": [...]}`;
every feature has a unique `id` and a `type`.

### Sketch

```json
{"id": "S1", "type": "sketch",
 "plane": {"origin": [0,0,0], "normal": [0,0,1], "x_dir": [1,0,0]},
 "profiles": [{"outer": [<curves>], "inner": [[<curves>], ...]}]}
```

- 2-D coordinates are in the plane's local frame: local X = `x_dir`, local Y = `normal × x_dir`.
- Curves: `{"type":"line","start","end"}`, `{"type":"arc","start","mid","end"}` (3-point),
  `{"type":"circle","center","radius"}`. A circle must be alone in its loop; other loops must be closed and
  continuous (each curve's `end` = the next curve's `start`).
- `inner` loops are islands removed from the profile.

### Sketch-based features

| type | fields | notes |
|------|--------|-------|
| `pad` | `sketch`, `length`, `direction` (±1, default 1), `symmetric` | adds material; direction is relative to sketch normal |
| `pocket` | `sketch`, `depth` **or** `through_all`, `direction` (±1, default −1) | removes material |

### Hole

```json
{"id": "H1", "type": "hole", "kind": "simple|counterbore|countersink",
 "position": [x,y,z], "direction": [dx,dy,dz], "diameter": d,
 "depth": h | "through_all": true,
 "cb_diameter": D, "cb_depth": t,          // counterbore only
 "cs_diameter": D, "cs_angle": 90}         // countersink only
```

`position` lies on the entry face; `direction` points into the material. Flat-bottomed.

### Dress-up: fillet / chamfer

```json
{"id": "F1", "type": "fillet",  "radius": 5,   "edges": [{"point": [10,30,10], "curve": "line"}]}
{"id": "C1", "type": "chamfer", "distance": 1, "edges": [{"point": [50,0,0],   "curve": "line"}]}
```

**Edge references** are topology-independent: a point lying on the edge plus the edge's
curve type (`line|circle|ellipse|bspline|other`). A rebuilder picks the nearest edge of matching
type in the *current* body (tolerance 1e-3 mm for exact JSON; `QUANTIZED_EDGE_TOL` = 0.3 mm for
tokenized / predicted sequences, e.g. `build(seq, edge_tol=QUANTIZED_EDGE_TOL)`). This is what makes
the history replayable in a different kernel (CATIA) where edge IDs differ.

### Mirror

```json
{"id": "M1", "type": "mirror", "plane": {"origin": [0,0,0], "normal": [0,0,1]}}
```

Mirrors the whole current body and fuses it with the original (CATIA: *Mirror* of PartBody).

## Canonical ordering (generator rules)

Several histories can produce the same solid. To reduce that ambiguity for training, generated
sequences follow one fixed order:

1. Sketch + pad features (the first pad creates the body)
2. Sketch + pocket features
3. Holes
4. Fillets, then chamfers
5. Mirror (last — so dress-ups are mirrored for free)

Each sketch is immediately followed by the pad/pocket that consumes it.

## Synthetic dataset generator (`datagen/`)

```powershell
# full run (resumable: re-running skips finished ids)
myenv\Scripts\python.exe -m datagen.run --out data\synth_v2 --count 20000
# contact sheet of random samples
myenv\Scripts\python.exe -m datagen.preview data\synth_v2 --n 24 --out preview.png
```

Output per sample (`<shard>` = id without its last 3 digits, 1000 files per folder):

| Path | Content |
|------|---------|
| `step/<shard>/<id>.step` | dumb solid — model **input** |
| `json/<shard>/<id>.json` | design history — model **target** |
| `graph/<shard>/<id>.npz` | UV-Net arrays `node_x (F,10,10,7)`, `edge_index (2,E)`, `edge_x (E,10,6)`; load with `cadseq.graph.to_dgl` |
| `index.jsonl` / `summary.json` | per-sample metadata (family, op counts, faces, seed…) / dataset statistics |

### How a sample is generated

1. **Base family** (`families.py`), one of: prismatic — `plate` (rect / rounded / chamfered / polygon),
   `block`, `l_bracket` (optionally rounded top), `gusset_bracket`, `u_channel`, `t_block`;
   turned — `stepped_shaft`, `flange` (hub, optional spigot, bore), `hub_disc`.
2. **Decoration stages** (`features.py`) in canonical order, each with a random count:
   bosses → pockets (rect / rounded rect / circle / slot / polygon; blind or through) →
   hole groups (single / linear / grid / circular pattern; simple, counterbore or countersink with
   metric M3–M16 sizes; blind or through) → fillets → chamfers → optional mirror.
3. Features are only placed on **real axis-aligned planar faces** of the current body; a profile
   must fit inside the face with a margin, pocket/hole depths come from the measured wall thickness,
   and fillet/chamfer edges are chosen from actual edges (sizes ≤ 30% of the shortest selected edge).
4. Each feature group is applied **atomically** — kernel errors, invalid solids, no-op features,
   sliver geometry (any edge < 0.2 mm) or > 200 faces roll it back.
5. All geometric values are **snapped to the tokenizer grid (0.05 mm)**, so the stored history is
   exactly representable as tokens.

### Validation (a sample is kept only if all pass)

- The history is canonicalized through the tokenizer (`decode(encode(seq))`); the canonical JSON is
  what gets saved, and replaying it from scratch must reproduce the incremental body (volume, faces).
- STEP export → re-import gives one valid solid with identical faces/volume.
- UV-Net graph has no NaN/inf and the face graph is connected.

`datagen.run` supervises its own worker processes: a worker that crashes inside OpenCascade or
exceeds `--timeout` is killed and restarted, and the sample is retried with a fresh seed range.

Sequences are deterministic per `(--seed, id)`; use a different `--seed` for a disjoint split.

## Tokenizer (`cadseq/tokenizer.py`)

`encode(seq) -> list[int]`, `decode(ids) -> CADSequence`, `to_text(ids)` for debugging.
Vocabulary: 335 tokens = 3 special + 30 structural + 102 coarse + 200 fine number tokens.

**Numbers** are quantized to 0.05 mm (angles to 0.05°) in the range [-510, 510) and written as
two tokens: a 10 mm bucket `#k` and a 0.05 mm offset `.k` (value = (k_coarse·200 + k_fine − 10200) / 20).

**Grammar** (each feature starts with a feature token; no ids or sketch references are emitted):

```
SKETCH axis n_offset ( PROFILE LOOP curve+ (LOOP curve+)* )+       curve: LINE x y | ARC x y mx my | CIRCLE cx cy r
PAD    (EXT+|EXT-|EXT_SYM) n_length                                 # consumes the preceding SKETCH
POCKET (EXT+|EXT-) (THROUGH | n_depth)                              # consumes the preceding SKETCH
HOLE   (H_SIMPLE | H_CBORE D t | H_CSINK D angle) axis x y z d (THROUGH | n_depth)
FILLET r  (E_LINE|E_CIRCLE|...) x y z ...       CHAMFER d (E_LINE|E_CIRCLE|...) x y z ...
MIRROR (+X|+Y|+Z) n_offset
```

Example (`examples/bracket.json`, 184 tokens; `to_text` output):

```
SKETCH -Y 60 PROFILE LOOP LINE 0 0 LINE 100 0 LINE 100 10 LINE 10 10 ARC 10 70 5 75 LINE 0 70
PAD EXT+ 60
SKETCH +Z 10 PROFILE LOOP LINE 55 24 ARC 85 24 91 30 LINE 85 36 ARC 55 36 49 30
POCKET EXT- THROUGH
HOLE H_CBORE 11 4 +X 0 15 45 6.6 THROUGH
HOLE H_CSINK 10 90 -Z 25 15 10 5.5 THROUGH
FILLET 8 E_LINE 100 0 5 E_LINE 100 60 5
CHAMFER 1 E_LINE 50 0 0
```

**Canonicalization** (removes ambiguity the model would otherwise have to guess):
sketch planes must be axis-aligned with the origin on the axis and the canonical `x_dir`
(+Y for ±X normals, +X otherwise); lines/arcs store only their start point (the end is the next
curve's start); loops are re-oriented counter-clockwise and start at the lowest (x, y) vertex;
inner loops, profiles and fillet/chamfer edges are sorted; ids are regenerated (S1, P1, K1, H1…).
Inputs that cannot be represented raise `TokenizeError`; ungrammatical / invalid token sequences
raise `DecodeError` (use this to reject bad model outputs).

## Training data (`cadseq/prepare.py`, `cadseq/dataset.py`)

```powershell
# once per dataset: tokens.npz, graphs/ (packed, memory-mapped), splits.json (90/5/5 by family), vocab.json
.venv\Scripts\python.exe -m cadseq.prepare data\synth_v2
```

```python
from torch.utils.data import DataLoader
from cadseq.dataset import CADGraphSeqDataset, LengthBucketSampler, collate

ds = CADGraphSeqDataset("data/synth_v2", split="train")          # also "val" / "test"
dl = DataLoader(ds, batch_sampler=LengthBucketSampler(ds, 32), collate_fn=collate)
batch = next(iter(dl))
batch["node_x"]      # (ΣF,10,10,7)  face UV-grids
batch["edge_x"]      # (ΣE,10,6)     edge U-grids (directed)
batch["edge_index"]  # (2,ΣE)        global face ids;  batch["node_batch"] (ΣF,) sample per face
batch["tokens"]      # (B, L) int64, <bos> … <eos>, padded with PAD_ID = 0
```

Graphs are memory-mapped from `graphs/*.npy`, so `num_workers=0` is fast (no Windows worker
start-up). For a DGLGraph of one sample use `cadseq.graph.to_dgl` on its `.npz` file.

Point coordinates in the grids are scaled by 0.01 (1 unit = 100 mm) but **not centred**: the
target tokens are absolute mm in the world frame, so the encoder sees the same frame. (Real STEP
files will need a placement canonicalization step before inference — planned for step 6.)

## Model (`s2pnet/`)

```
face grids (ΣF,10,10,7) ─ SurfaceCNN (UV-Net 2-D conv) + geometric summary MLP ─┐
edge grids (ΣE,10,6)    ─ CurveCNN   (UV-Net 1-D conv) + endpoint/tangent MLP ──┤
                                                                                ▼
                       4 × edge-aware message passing (faces ↔ shared edges, edges updated too)
                                                                                ▼
                       [GLOBAL] + face tokens → 3-layer Transformer encoder  ──► memory (B, 1+Fmax, 384)
                                                                                ▼ cross-attention
<bos> … ─► 8-layer pre-norm Transformer decoder (d=384, 8 heads, tied embeddings, KV cache) ─► next token
```

- **Encoder** (`encoder.py`): the geometric summary per face (masked centroid, bbox, mean normal,
  trimmed fraction) gives the network direct access to absolute positions, which the tokens need.
- **Decoder** (`decoder.py`): teacher-forced cross-entropy for training; `generate()` does greedy
  or sampled decoding with a KV cache.
- **Grammar-constrained decoding** (`cadseq/grammar.py`): at every step only tokens allowed by the
  format are kept (feature order, number = coarse+fine pair, positive lengths/radii, ≥ 2 curves per
  loop, …). Every one of the 20K dataset sequences is accepted by the grammar.

## Training & evaluation

```powershell
# train (resumes automatically from runs\base\last.pt; logs runs\base\log.jsonl; best.pt = lowest val loss)
.venv\Scripts\python.exe -m s2pnet.train --data data\synth_v2 --out runs\base --epochs 60
# CAD-level evaluation: generate → rebuild with CadQuery → compare with the input STEP
.venv\Scripts\python.exe -m s2pnet.evaluate --ckpt runs\base\best.pt --split test --n 1000
```

Defaults: AdamW (lr 3e-4, wd 0.05, β = 0.9/0.98), 1000 warm-up steps, cosine decay, bf16 autocast,
batch 32 with length bucketing, grad-clip 1.0. ~37.6M parameters, ~24K tokens/s, ~3.5 min/epoch on
the RTX PRO 2000 (8 GB; peak ≈ 7.8 GB — use `--bs 24` if you hit out-of-memory).

During training (`log.jsonl`): `val_loss`, `val_tok_acc` (teacher forcing) and, every `--gen-every`
epochs, grammar-constrained greedy generation on `--gen-samples` val parts: `gen_exact` (identical
tokens), `gen_valid` (decodes), `gen_feature_types_match`.

`evaluate.py` metrics (CAD kernel in parallel worker processes):

| metric | meaning |
|--------|---------|
| `exact` | predicted tokens identical to the ground truth |
| `valid` | tokens decode into a valid `CADSequence` |
| `built_strict` | the history rebuilds exactly as predicted (edge refs within 0.3 mm) |
| `built` | rebuilds with `build_lenient` (see below) |
| `iou_*` | Monte Carlo volume IoU of the rebuild vs the input STEP (8192 points, ≈ 0.5 % std. error; 0 if not built) |
| `iou_strict_mean` | IoU counting only strict rebuilds |
| `chamfer_median_built` | surface Chamfer distance / bbox diagonal; ≈ 0.009 is the sampling floor for identical shapes |
| `mean_skipped_features` | features dropped by the lenient rebuild |

**Lenient rebuild** (`cadseq.build.build_lenient`, used by `evaluate.py` and `predict.py` when the
strict rebuild fails): predicted fillet/chamfer edge points are typically 1–4 mm off the true edge,
which fails the strict 0.3 mm match and — because one failed feature kills the whole part — made
most predictions unbuildable. The lenient builder snaps each edge ref to the nearest edge of the
same curve type within 3 % of the body's bbox diagonal (dropping duplicates/unresolvable refs) and
skips any feature that still fails. Mid-training (epoch 25, 100 val parts) this raised the build
rate from 12 % to 94 %.

IoU is computed by point classification (`cadseq/metrics.py`) rather than OCC booleans, which fail
on near-coincident solids (prediction vs STEP round-trip) and returned IoU = 0 for exact matches.

Sanity check that was run: the model memorises 64 training parts to 100 % exact generation and
IoU 1.0 (`--limit 64 --val-split train --dropout 0`).

## Results — `runs/base` (60 epochs, synth_v2)

Held-out **test split, 1000 parts**, grammar-constrained greedy decoding, lenient rebuild:

| checkpoint | built (strict / lenient) | IoU mean | IoU median | IoU > 0.9 | > 0.95 | > 0.99 | Chamfer median |
|---|---|---|---|---|---|---|---|
| `last.pt` (epoch 59) | 15.1 % / **94.7 %** | **0.897** | **0.970** | **84.6 %** | **70.6 %** | **13.9 %** | 0.0093 |
| `best.pt` (lowest val loss, epoch 34) | 14.3 % / 93.8 % | 0.873 | 0.954 | 81.5 % | 54.4 % | 4.3 % | 0.0094 |

100 % of predictions are grammatical and decode; 0 % are token-for-token identical (sequences
are ~260 tokens of 0.05 mm numbers). Use **`last.pt`**: validation *loss* rises after epoch ~30
(overfitting / over-confidence; train loss 0.15 vs val 0.49) but generation quality keeps improving.

Per family (`last.pt`): turned parts are easiest (flange / hub_disc median IoU 0.984, 30–35 % > 0.99);
plates, blocks, u-channels and stepped shafts are hardest (median 0.95–0.96).

Remaining errors: strict rebuilds fail almost entirely on fillet/chamfer **edge references**
(predicted points 1–4 mm off), and IoU is capped by slightly wrong **dimensions/positions** of
pockets and holes. 5.3 % of parts produce no valid body at all.

Out-of-distribution check — `examples/SimpleRectangle.stp` (20 × 122.1 × 88.8 mm,
centred on Y/Z): the model predicts a ~40 × 122.5 × 90 box built up from z = 0 (IoU 0.21). Sizes are
roughly recognised but the **placement convention differs** from the synthetic data, which always
starts at z = 0 with sketch planes through the origin → placement canonicalization is needed (step 6).

```powershell
.venv\Scripts\python.exe -m s2pnet.predict --ckpt runs\base\last.pt --step part.stp --out out\part
```

Training notes: the RTX PRO 2000 laptop GPU throttles thermally (~89 °C, 915 of 3090 MHz) which halves
throughput (≈ 48K → 24K tokens/s, 110 → 225 s/epoch). `train.py` blocks Windows idle sleep / Modern
Standby (which once paused a run for 75 min); closing the lid still suspends it.
