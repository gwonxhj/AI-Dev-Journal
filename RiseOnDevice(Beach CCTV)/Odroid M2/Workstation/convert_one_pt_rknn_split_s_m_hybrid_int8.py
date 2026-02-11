#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import onnx
from onnx import helper, TensorProto

from rknn.api import RKNN

# =========================
# 안정화(PC 리부트 방지용) - 스레드 폭주 억제
# (주의) 반드시 "import numpy/onnx/rknn" 보다 위에서 세팅하는 게 가장 좋지만
#       이미 작성된 흐름 유지하되, 여기서라도 강제.
# =========================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# 추가: 파이썬 해시/스레드 등 흔들림 최소화 (선택)
os.environ.setdefault("PYTHONHASHSEED", "0")

TARGET_PLATFORM = "rk3588"
IMGSZ = 640
OPSET = 12

RKNN_CONFIG = dict(
    target_platform=TARGET_PLATFORM,
    optimization_level=3,
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    quantized_algorithm="normal",   # mmse는 리부트 이슈 → normal로 고정
    quantized_method="channel",
)

# =========================
# Preset B (네가 말한 "balanced" 확장판 = 내가 추천한 Preset B)
# - 기존 7개 + box 산술 라인 8개
# =========================
PRESET_B_KEYS = [
    # outputs
    "output0_boxes",
    "output0_scores",

    # DFL 핵심
    "/model.22/dfl/Softmax_output_0_sw_sw",
    "/model.22/dfl/Transpose_1_output_0_sw",
    "/model.22/dfl/Reshape_1_output_0_rs",
    "/model.22/dfl/Reshape_output_0",

    # box 산술
    "/model.22/Mul_2_output_0-rs",

    # box 산술 라인 추가(정확도 복구용)
    "/model.22/Concat_2_output_0-rs",
    "/model.22/Div_1_output_0-rs",
    "/model.22/Add_2_output_0-rs",
    "/model.22/Sub_1_output_0-rs",
    "/model.22/Add_1_output_0-rs",
    "/model.22/Sub_output_0-rs",
    "/model.22/Slice_output_0-rs",
    "/model.22/Slice_1_output_0-rs",
]

# 네 기존 프리셋도 유지(원하면 선택 가능)
FP16_PRESETS = {
    "A": [  # 최소(기존 accurate였던 7개)
        "output0_boxes",
        "output0_scores",
        "/model.22/dfl/Softmax_output_0_sw_sw",
        "/model.22/dfl/Transpose_1_output_0_sw",
        "/model.22/dfl/Reshape_1_output_0_rs",
        "/model.22/dfl/Reshape_output_0",
        "/model.22/Mul_2_output_0-rs",
    ],
    "B": PRESET_B_KEYS,  # ✅ 추천
    "C": [  # aggressive (score만)
        "output0_scores",
    ],
}


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except Exception:
        return 0.0


@dataclass
class Step1Outputs:
    cfg: Path
    model: Path
    data: Path


def find_latest_step1_outputs(search_dirs: List[Path], prefix_hint: Optional[str] = None) -> Step1Outputs:
    """
    step1 산출물이 convert/든 _hybrid_tmp/든 상관없이
    최신 cfg 기준으로 같은 stem의 model/data가 있는 세트를 찾아 반환
    """
    cfgs, models, datas = [], [], []
    for d in search_dirs:
        if not d.exists():
            continue
        cfgs += list(d.glob("*.quantization.cfg"))
        models += list(d.glob("*.model"))
        datas += list(d.glob("*.data"))

    if prefix_hint:
        cfgs_hint = [p for p in cfgs if p.name.startswith(prefix_hint)]
        models_hint = [p for p in models if p.name.startswith(prefix_hint)]
        datas_hint = [p for p in datas if p.name.startswith(prefix_hint)]
        if cfgs_hint and models_hint and datas_hint:
            cfgs, models, datas = cfgs_hint, models_hint, datas_hint

    if not cfgs or not models or not datas:
        raise RuntimeError(
            f"[STEP1] outputs not found.\n"
            f" searched: {', '.join(str(d) for d in search_dirs)}\n"
            f" cfg={len(cfgs)} model={len(models)} data={len(datas)}"
        )

    cfgs_sorted = sorted(cfgs, key=_mtime, reverse=True)
    for cfg in cfgs_sorted:
        stem = cfg.name.replace(".quantization.cfg", "")
        cand_model = [p for p in models if p.name == f"{stem}.model"]
        cand_data  = [p for p in datas  if p.name == f"{stem}.data"]
        if cand_model and cand_data:
            return Step1Outputs(cfg=cfg, model=cand_model[0], data=cand_data[0])

    # fallback
    model_latest = sorted(models, key=_mtime, reverse=True)[0]
    data_latest  = sorted(datas, key=_mtime, reverse=True)[0]
    cfg_latest   = cfgs_sorted[0]
    return Step1Outputs(cfg=cfg_latest, model=model_latest, data=data_latest)


def filter_keys_exist_in_cfg(cfg_path: Path, keys: List[str]) -> List[str]:
    """
    ✅ 요청한 기능:
    - step1 cfg(quantize_parameters 섹션)에 실제 존재하는 키만 선별
    - 존재 기준: 줄 시작에 '    <key>:' 또는 '<key>:' 형태가 있으면 OK
    """
    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    # 빠른 membership 검색을 위해 앞에 '\n' 붙여서 라인 매칭 안정화
    t = "\n" + text

    valid = []
    for k in keys:
        # quantize_parameters 쪽은 보통 '    <key>:' 형태
        if f"\n    {k}:" in t or f"\n{k}:" in t:
            valid.append(k)
    return valid


def patch_cfg_text_only(cfg_path: Path, fp16_keys: List[str]) -> Path:
    """
    YAML load/dump 금지(quantize_parameters 손대면 step2 에러 가능)
    - custom_quantize_layers: {} => 블록으로 교체
    - custom_quantize_layers 블록에 fp16만 삽입
    - cfg.orig / cfg.patched 생성
    """
    orig_text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    lines = orig_text.splitlines(True)

    # quantize_parameters 위치
    qp_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == "quantize_parameters:":
            qp_idx = i
            break
    if qp_idx is None:
        raise RuntimeError("quantize_parameters: not found in cfg")

    # custom_quantize_layers 위치(없으면 생성)
    cql_idx = None
    for i, ln in enumerate(lines[:qp_idx]):
        if ln.strip().startswith("custom_quantize_layers:"):
            cql_idx = i
            break
    if cql_idx is None:
        lines.insert(qp_idx, "custom_quantize_layers:\n")
        cql_idx = qp_idx
        qp_idx += 1

    # custom_quantize_layers: {} 형태면 블록으로 교체
    if lines[cql_idx].strip() == "custom_quantize_layers: {}":
        lines[cql_idx] = "custom_quantize_layers:\n"

    # 이미 있는 키 수집(중복 삽입 방지)
    existing = set()
    for ln in lines[cql_idx + 1: qp_idx]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if ":" in s:
            key = s.split(":", 1)[0].strip()
            existing.add(key)

    add_lines = []
    for k in fp16_keys:
        if k in existing:
            continue
        add_lines.append(f"    {k}: float16\n")

    # 보기 좋게 정렬 (원하면 제거 가능)
    add_lines = sorted(add_lines)
    if add_lines:
        lines[cql_idx + 1: cql_idx + 1] = add_lines

    # orig/patched 저장 (항상 남김)
    cfg_orig = cfg_path.with_suffix(cfg_path.suffix + ".orig")
    cfg_orig.write_text(orig_text, encoding="utf-8")

    patched_path = cfg_path.with_suffix(cfg_path.suffix + ".patched")
    patched_path.write_text("".join(lines), encoding="utf-8")
    return patched_path


def export_pt_to_onnx(pt_path: Path, onnx_path: Path, imgsz=IMGSZ, opset=OPSET) -> None:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(f"ultralytics import failed: {repr(e)} (pip install ultralytics)")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(pt_path))
    model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False, simplify=False)

    cand = []
    for d in [pt_path.parent, Path.cwd(), onnx_path.parent]:
        cand += list(d.glob("*.onnx"))
    if not cand:
        raise RuntimeError("PT -> ONNX export failed: no onnx file created")

    newest = sorted(cand, key=_mtime, reverse=True)[0]
    shutil.copy2(newest, onnx_path)


def split_onnx_output(onnx_in: Path, onnx_out: Path) -> Tuple[int, int]:
    """
    output0 [1,C,N] 또는 [1,N,C]를
    output0_boxes (4) + output0_scores (C-4)로 split.
    """
    m = onnx.load(str(onnx_in))
    if len(m.graph.output) != 1:
        onnx.save(m, str(onnx_out))
        return (1, 8400)

    out0 = m.graph.output[0]
    name0 = out0.name
    dims = [d.dim_value for d in out0.type.tensor_type.shape.dim]
    if len(dims) != 3:
        raise RuntimeError(f"Unexpected output dims={dims} (need 3D)")

    d1, d2 = dims[1], dims[2]
    if d1 <= 512 and d2 > d1:
        C, N = d1, d2
        layout = "CN"
    elif d2 <= 512 and d1 > d2:
        N, C = d1, d2
        layout = "NC"
    else:
        C, N = d1, d2
        layout = "CN"

    if C < 5:
        raise RuntimeError(f"Output channels too small: C={C}, expected >=5")

    axis = 1 if layout == "CN" else 2

    def add_const(name: str, vals: List[int]) -> str:
        t = helper.make_tensor(name=name, data_type=TensorProto.INT64, dims=[len(vals)], vals=vals)
        m.graph.initializer.append(t)
        return name

    starts_boxes = add_const("starts_boxes", [0])
    ends_boxes   = add_const("ends_boxes", [4])
    axes_boxes   = add_const("axes_boxes", [axis])
    steps_boxes  = add_const("steps_boxes", [1])

    starts_scores = add_const("starts_scores", [4])
    ends_scores   = add_const("ends_scores", [C])
    axes_scores   = add_const("axes_scores", [axis])
    steps_scores  = add_const("steps_scores", [1])

    boxes_name = "output0_boxes"
    scores_name = "output0_scores"

    node_boxes = helper.make_node("Slice", [name0, starts_boxes, ends_boxes, axes_boxes, steps_boxes], [boxes_name], name="Slice_boxes")
    node_scores = helper.make_node("Slice", [name0, starts_scores, ends_scores, axes_scores, steps_scores], [scores_name], name="Slice_scores")
    m.graph.node.extend([node_boxes, node_scores])

    m.graph.output.pop(0)
    if layout == "CN":
        boxes_shape = [1, 4, N]
        scores_shape = [1, C - 4, N]
    else:
        boxes_shape = [1, N, 4]
        scores_shape = [1, N, C - 4]

    m.graph.output.extend([
        helper.make_tensor_value_info(boxes_name, TensorProto.FLOAT, boxes_shape),
        helper.make_tensor_value_info(scores_name, TensorProto.FLOAT, scores_shape),
    ])

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(onnx_out))
    return (C - 4, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=str(Path.home() / "convert"))
    ap.add_argument("--model", default="s", choices=["n", "s", "m", "l"],
                    help="which model to convert: n/s/m/l (expects models/yolov8?.pt)")
    ap.add_argument("--preset", default="B", choices=["A", "B", "C"],
                    help="A=7keys, B=PresetB(추천), C=score-only")
    ap.add_argument("--auto_keys", action="store_true",
                    help="filter FP16_KEYS by cfg existence (recommended for S/M/L)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    dataset_txt = base_dir / "dataset.txt"
    pt = base_dir / "models" / f"yolov8{args.model}.pt"
    out_dir = base_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_txt.exists():
        raise FileNotFoundError(f"dataset.txt not found: {dataset_txt}")
    if not pt.exists():
        raise FileNotFoundError(f"pt not found: {pt}")

    stem = f"yolov8{args.model}"
    onnx_fp32 = out_dir / f"{stem}_fp32.onnx"
    onnx_split = out_dir / f"{stem}_fp32_split.onnx"
    out_rknn = out_dir / f"{stem}_hybrid_int8_preset{args.preset}.rknn"
    work_dir = base_dir / f"_hybrid_tmp_{stem}_preset{args.preset}"

    print(f"[INFO] PT      : {pt}")
    print(f"[INFO] DATASET : {dataset_txt}")
    print(f"[INFO] PRESET  : {args.preset} (auto_keys={args.auto_keys})")
    print(f"[INFO] OUTDIR  : {out_dir}")
    print(f"[INFO] WORKDIR : {work_dir}")

    print("== PT -> ONNX ==")
    export_pt_to_onnx(pt, onnx_fp32)

    print("== ONNX split ==")
    nc, N = split_onnx_output(onnx_fp32, onnx_split)
    print(f"[INFO] inferred num_classes={nc}, anchors={N}")

    # -------------------------
    # Hybrid step1
    # -------------------------
    print("== Hybrid INT8 step1 ==")
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_txt, work_dir / "dataset.txt")

    rknn = RKNN(verbose=True)
    rknn.config(**RKNN_CONFIG)

    ret = rknn.load_onnx(model=str(onnx_split))
    if ret != 0:
        raise RuntimeError(f"load_onnx failed ret={ret}")

    cwd = Path.cwd()
    os.chdir(work_dir)
    try:
        ret = rknn.hybrid_quantization_step1(dataset="dataset.txt")
        if ret != 0:
            raise RuntimeError(f"step1 failed ret={ret}")
    finally:
        os.chdir(cwd)

    # step1 산출물 찾기(어디로 생성되든 OK)
    outs = find_latest_step1_outputs(
        search_dirs=[work_dir, base_dir, base_dir / "_hybrid_tmp", Path.cwd()],
        prefix_hint=f"{stem}_fp32_split",
    )
    print(f"[STEP1] cfg  : {outs.cfg}")
    print(f"[STEP1] model: {outs.model}")
    print(f"[STEP1] data : {outs.data}")

    # -------------------------
    # Patch cfg (Preset B 적용)
    # -------------------------
    print("== patch cfg ==")
    fp16_keys = FP16_PRESETS[args.preset]

    if args.auto_keys:
        fp16_keys2 = filter_keys_exist_in_cfg(outs.cfg, fp16_keys)
        print(f"[AUTO_KEYS] requested={len(fp16_keys)} -> exists_in_cfg={len(fp16_keys2)}")
        for k in fp16_keys2:
            print("  fp16:", k)
        fp16_keys = fp16_keys2
        if not fp16_keys:
            raise RuntimeError("[AUTO_KEYS] no fp16 keys matched in cfg. Check cfg names/stem.")
    else:
        print(f"[FP16_KEYS] count={len(fp16_keys)}")
        for k in fp16_keys:
            print("  fp16:", k)

    patched_cfg = patch_cfg_text_only(outs.cfg, fp16_keys)
    print(f"[PATCH] saved: {patched_cfg}")
    print(f"[PATCH] orig : {outs.cfg.with_suffix(outs.cfg.suffix + '.orig')}")

    # -------------------------
    # Hybrid step2
    # -------------------------
    print("== Hybrid INT8 step2 ==")
    ret = rknn.hybrid_quantization_step2(
        model_input=str(outs.model),
        data_input=str(outs.data),
        model_quantization_cfg=str(patched_cfg),
    )
    if ret != 0:
        raise RuntimeError(f"step2 failed ret={ret}")

    # -------------------------
    # export rknn
    # -------------------------
    print("== export rknn ==")
    ret = rknn.export_rknn(str(out_rknn))
    if ret != 0:
        raise RuntimeError(f"export_rknn failed ret={ret}")

    print(f"[OK] RKNN saved: {out_rknn}")
    print(f"[OK] Patched cfg kept: {patched_cfg}")


if __name__ == "__main__":
    main()
