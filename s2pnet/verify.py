"""Generate-and-verify: sample several candidate histories, rebuild each, keep the best match.

Selection compares each rebuilt candidate with the **input** solid (the only thing available at
inference), so it is legitimate at test time. Candidates are ranked by Monte Carlo IoU against the
input (on ``select_points`` points, seeded differently from the final metric to avoid selection
bias), then by strict-rebuild success.

    cands = generate_candidates(model, batch, samples=8, temperature=0.8)   # token lists per item
    best = select_best([decode(c) for c in cands[0]], input_shape)          # in-process
    res  = cad_check((cand_jsons, step_path, 2048, True))                   # worker-process entry
"""

from __future__ import annotations

import warnings

from cadseq.tokenizer import DecodeError, decode

SELECT_SEED = 1  # final metrics use seed 0 (cadseq.metrics default)
EARLY_STOP_IOU = 0.995


def generate_candidates(model, batch, samples: int = 1, temperature: float = 0.8, seed: int = 0,
                        amp: bool = True) -> list[list[list[int]]]:
    """Per batch item: unique token sequences, greedy first, then up to ``samples - 1`` sampled ones."""
    import torch  # lazy: CAD worker processes import this module without needing torch

    device = batch["tokens"].device
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        mem, pad = model.encode(batch)
        copy = model.copy_context(batch, mem, pad) if hasattr(model, "copy_context") else None
        out = [[seq] for seq in model.decoder.generate(mem, pad, copy=copy)]
        gen = torch.Generator(device=device)
        for k in range(1, samples):
            gen.manual_seed(seed * 1000 + k)
            for i, seq in enumerate(model.decoder.generate(mem, pad, temperature=temperature, generator=gen, copy=copy)):
                if seq not in out[i]:
                    out[i].append(seq)
    return out


def decode_candidates(cands: list[list[int]]) -> tuple[list[int], list]:
    """Decode token sequences, dropping ungrammatical ones. Returns (kept indices, CADSequences)."""
    idx, seqs = [], []
    for i, c in enumerate(cands):
        try:
            seqs.append(decode(c))
            idx.append(i)
        except DecodeError:
            pass
    return idx, seqs


def rebuild(seq):
    """Strict rebuild, falling back to the lenient one.

    Returns (body | None, info, history) where ``history`` is the sequence that strictly rebuilds
    ``body``: ``seq`` itself, or the lenient builder's repaired sequence.
    """
    from cadseq.build import QUANTIZED_EDGE_TOL, build, build_lenient

    info = {"built_strict": False, "built": False, "skipped": 0}
    try:
        body = build(seq, edge_tol=QUANTIZED_EDGE_TOL)
        info["built_strict"] = info["built"] = True
        return body, info, seq
    except Exception as e:
        info["error"] = str(e)[:200]
    try:
        body, skipped, repaired = build_lenient(seq)
        info["built"] = True
        info["skipped"] = len(skipped)
        return body, info, repaired
    except Exception as e:
        info["error_lenient"] = str(e)[:200]
        return None, info, None


def select_best(seqs: list, target, select_points: int = 2048):
    """Rebuild every candidate and pick the best against ``target``.

    Returns (best index | None, best body | None, best history | None, per-candidate infos).
    With one candidate no selection IoU is computed.
    """
    from cadseq.metrics import mc_iou

    infos, best, best_key, best_body, best_hist = [], None, None, None, None
    for i, seq in enumerate(seqs):
        body, info, hist = rebuild(seq)
        if body is not None and len(seqs) > 1:
            try:
                info["iou_select"] = mc_iou(body, target, n=select_points, seed=SELECT_SEED)
            except Exception as e:
                info["error_metric"] = str(e)[:200]
                info["iou_select"] = 0.0
        infos.append(info)
        if body is None:
            continue
        key = (info.get("iou_select", 1.0), info["built_strict"], -i)
        if best_key is None or key > best_key:
            best, best_key, best_body, best_hist = i, key, body, hist
        if info["built_strict"] and info.get("iou_select", 0.0) >= EARLY_STOP_IOU:
            break  # cannot meaningfully improve on a strict, near-exact candidate
    return best, best_body, best_hist, infos


def cad_check(args) -> dict:
    """Worker-process entry. ``args = (candidate JSON strings, target STEP path, select_points, chamfer)``.

    Picks the best candidate against the target and reports final metrics for it.
    """
    warnings.filterwarnings("ignore")
    import cadquery as cq

    from cadseq.metrics import chamfer, mc_iou
    from cadseq.schema import CADSequence

    cand_jsons, step_path, select_points, with_chamfer = args
    seqs = [CADSequence.model_validate_json(j) for j in cand_jsons]
    res = {"built_strict": False, "built": False, "iou": 0.0, "iou_strict": 0.0, "skipped": 0,
           "chosen": None, "n_candidates": len(seqs), "n_rebuilt": 0}
    try:
        target = cq.importers.importStep(step_path).val()
    except Exception as e:
        res["error"] = f"target: {str(e)[:200]}"
        return res
    best, body, _, infos = select_best(seqs, target, select_points)
    res["n_rebuilt"] = sum(i["built"] for i in infos)
    if best is None:
        res["error"] = infos[0].get("error_lenient") or infos[0].get("error") if infos else "no candidates"
        return res
    info = infos[best]
    res.update({"chosen": best, "built_strict": info["built_strict"], "built": True, "skipped": info["skipped"]})
    if not info["built_strict"]:
        res["error"] = info.get("error")
    try:
        res["iou"] = mc_iou(body, target)
        if with_chamfer:
            res["chamfer"] = chamfer(body, target)
        res["faces"], res["gt_faces"] = len(body.Faces()), len(target.Faces())
        if info["built_strict"]:
            res["iou_strict"] = res["iou"]
    except Exception as e:
        res["error_metric"] = str(e)[:200]
    return res
