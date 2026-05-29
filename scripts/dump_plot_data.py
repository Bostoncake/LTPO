"""Dump all numeric arrays that appear in mathvista_r1r2_lookthink_preview.png
so they can be inlined into a self-contained reproduction script.
"""
from pathlib import Path
import numpy as np
import re

ROOT = Path("/home/xiongyizhe/research/LTPO")
OUT = ROOT / "output/visual_lookthink_r1r2"
npz = np.load(OUT / "mathvista_r1r2_curated.npz")

curated_qids = npz["curated_qids"].tolist()
with_look = npz["with_look"]  # (N, 10) running-min already? actually raw
no_look = npz["no_look"]
early_pick = int(npz["early_pick"][0])
late_pick = int(npz["late_pick"][0])

print("# curated_qids:", curated_qids)
print("# early_pick =", early_pick)
print("# late_pick  =", late_pick)
print("# arr shape  =", with_look.shape, no_look.shape)


def running_min(a):
    o = a.copy()
    for t in range(1, a.shape[-1]):
        o[..., t] = np.minimum(o[..., t - 1], a[..., t])
    return o


rm_with = running_min(with_look)
rm_no = running_min(no_look)

mean_with = rm_with.mean(axis=0)
std_with = rm_with.std(axis=0)
mean_no = rm_no.mean(axis=0)
std_no = rm_no.std(axis=0)

print("mean_with =", np.round(mean_with, 6).tolist())
print("std_with  =", np.round(std_with, 6).tolist())
print("mean_no   =", np.round(mean_no, 6).tolist())
print("std_no    =", np.round(std_no, 6).tolist())

# Now extract typical-sample trajectories + modes from the raw logs.
WITH_LOOK_LOG = (
    ROOT
    / "output/ltpo_dmlr_final/0518_param_search_perds_init_lookthink_stag_sigma_step"
    / "lookthink_stag_rewardentropy_diff_bestdiff_steps10_sigma25.0_decay0.95_topk10"
    / "math_vista_dev_tokens2_inithidden_lr1e-3_topp0.5_stag2.log"
)
NO_LOOK_LOG = (
    ROOT
    / "output/ltpo_dmlr_final/0516_param_search_perds_init_entropy"
    / "steps10_sigma25.0_decay0.95_lr1e-3_topk10_rewardentropy"
    / "math_vista_dev_tokens2_inithidden.log"
)

SAMPLE_HDR = re.compile(r"^\[(\d+)\] Q:")
WITH_STEP = re.compile(
    r">>> Step (\d+) entropy_diff: r1\(\+eps\)=([\-\d\.eE]+)\s+reward\(r1-r2\)=([\-\d\.eE]+)\s+mode=(\w+)"
)
NO_STEP = re.compile(
    r">>> Step (\d+) entropy r1\(\+eps\)=([\-\d\.eE]+)\s+r2\(-eps\)=([\-\d\.eE]+)"
)


def parse_with(path, target_qids):
    out = {}
    cur_qid = None
    cur = None
    for line in path.read_text().splitlines():
        m = SAMPLE_HDR.match(line)
        if m:
            if cur is not None and cur_qid in target_qids:
                out[cur_qid] = cur
            cur_qid = int(m.group(1))
            cur = {"r1mr2": [], "mode": []}
            continue
        m = WITH_STEP.search(line)
        if m and cur is not None:
            cur["r1mr2"].append(float(m.group(3)))
            cur["mode"].append(m.group(4))
    if cur is not None and cur_qid in target_qids:
        out[cur_qid] = cur
    return out


def parse_no(path, target_qids):
    out = {}
    cur_qid = None
    cur = None
    for line in path.read_text().splitlines():
        m = SAMPLE_HDR.match(line)
        if m:
            if cur is not None and cur_qid in target_qids:
                out[cur_qid] = cur
            cur_qid = int(m.group(1))
            cur = {"r1": [], "r2": []}
            continue
        m = NO_STEP.search(line)
        if m and cur is not None:
            cur["r1"].append(float(m.group(2)))
            cur["r2"].append(float(m.group(3)))
    if cur is not None and cur_qid in target_qids:
        out[cur_qid] = cur
    return out


targets = {early_pick, late_pick}
w = parse_with(WITH_LOOK_LOG, targets)
n = parse_no(NO_LOOK_LOG, targets)
for q in [early_pick, late_pick]:
    print(f"\n# qid {q}")
    print(f"with_look_r1mr2_{q} =", np.round(w[q]['r1mr2'], 6).tolist())
    print(f"with_look_mode_{q}  =", w[q]['mode'])
    no_r1mr2 = (np.array(n[q]['r1']) - np.array(n[q]['r2'])).round(6).tolist()
    print(f"no_look_r1mr2_{q}   =", no_r1mr2)
