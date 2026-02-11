import os
import shutil
from pathlib import Path

# rknn-toolkit2 2.3.2
from rknn.api import RKNN


# =========================
# USER CONFIG
# =========================
ONNX_PATH = Path("./yolov8n_fp32_split.onnx")   # 이미 split된 onnx 경로
DATASET_TXT = Path("./dataset.txt")            # convert 폴더에 있는 dataset.txt
OUT_RKNN = Path("./yolov8n_hybrid_int8_boxdfl_scorefix.rknn")

TARGET_PLATFORM = "rk3588"

# 전처리 기준: INT8 입력일 때 보통 mean=0 std=255 (0~255 -> 0~1)
RKNN_CONFIG = dict(
    target_platform=TARGET_PLATFORM,
    optimization_level=3,
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    quantized_algorithm="normal",   # MMSE는 PC 재부팅 이슈 있으니 일단 normal
    quantized_method="channel",
)

WORK_DIR = Path("./_hybrid_tmp")


# =========================
# TEXT ONLY CFG PATCH
# =========================
def patch_cfg_text_only(cfg_path: Path, fp16_keys):
    """
    TEXT ONLY patch:
    - YAML load/dump 금지 (quantize_parameters scale 같은 값 절대 변형 안함)
    - custom_quantize_layers: {} 형태면 -> custom_quantize_layers: 블록으로 교체
    - custom_quantize_layers 블록에만 "    key: float16" 추가
    - 기존 key는 중복 삽입 금지
    """
    lines = cfg_path.read_text(encoding="utf-8").splitlines(True)  # keep \n

    # 1) quantize_parameters 라인 찾기
    qp_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == "quantize_parameters:":
            qp_idx = i
            break
    if qp_idx is None:
        raise RuntimeError("quantize_parameters: not found in cfg")

    # 2) custom_quantize_layers 라인 찾기 (quantize_parameters 이전 영역에서)
    cql_idx = None
    for i, ln in enumerate(lines[:qp_idx]):
        if ln.strip().startswith("custom_quantize_layers:"):
            cql_idx = i
            break

    # 3) 없으면 quantize_parameters 바로 위에 생성
    if cql_idx is None:
        lines.insert(qp_idx, "custom_quantize_layers:\n")
        cql_idx = qp_idx
        qp_idx += 1

    # 4) custom_quantize_layers: {} 이면 블록으로 교체
    if lines[cql_idx].strip() == "custom_quantize_layers: {}":
        lines[cql_idx] = "custom_quantize_layers:\n"

    # 5) custom_quantize_layers 블록 범위: cql_idx+1 ~ qp_idx-1
    existing = set()
    for ln in lines[cql_idx + 1: qp_idx]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if ":" in s:
            key = s.split(":", 1)[0].strip()
            existing.add(key)

    # 6) 추가할 라인 만들기(중복 제거)
    add_lines = []
    for k in fp16_keys:
        if k in existing:
            continue
        add_lines.append(f"    {k}: float16\n")

    if not add_lines:
        print("== patch cfg (TEXT ONLY): nothing to add (already present) ==")
        return

    # 보기 좋게 정렬(원치 않으면 삭제 가능)
    add_lines = sorted(add_lines)

    # 7) custom_quantize_layers 바로 아래에 삽입
    lines[cql_idx + 1: cql_idx + 1] = add_lines

    cfg_path.write_text("".join(lines), encoding="utf-8")
    print(f"== patch cfg (TEXT ONLY): added {len(add_lines)} keys ==")


def pick_existing_keys_in_cfg(cfg_path: Path, candidates):
    """
    cfg 파일 안에 실제로 존재하는 key만 골라서 반환.
    (없거나 이름이 다른 operand 넣으면 step2에서 Invalid operands name 터짐 방지)
    """
    text = cfg_path.read_text(encoding="utf-8")
    found = []
    for k in candidates:
        # cfg 내부는 보통 "    <key>:" 형태로 존재
        if f"\n    {k}:" in ("\n" + text) or f"\n{k}:" in ("\n" + text):
            found.append(k)
    return found


# =========================
# FP16 KEY PRESETS
# =========================
def build_fp16_keys_boxdfl_scorefix(cfg_path: Path):
    """
    목표:
    - score 경로 float16 유지 (이미 너 scorefix 성공 경험 있음)
    - box/DFL 경로도 float16 유지해서 mAP 회복 시도
    - 단, cfg에 존재하는 키만 넣음(Invalid operands 방지)
    """
    candidates = [
        # scores
        "output0_scores",
        "output0_scores_int8",

        # boxes
        "output0_boxes",
        "output0_boxes_int8",

        # DFL + expectation path
        "/model.22/dfl/Softmax_output_0_sw_sw",
        "/model.22/dfl/Transpose_1_output_0_sw",
        "/model.22/dfl/Reshape_1_output_0_rs",
        "/model.22/dfl/Reshape_output_0",

        # box 산술 후처리(최소)
        "/model.22/Mul_2_output_0-rs",

        # (확장) box 정확도 개선 확률↑
        "/model.22/Concat_2_output_0-rs",
        "/model.22/Div_1_output_0-rs",
        "/model.22/Add_2_output_0-rs",
        "/model.22/Sub_1_output_0-rs",
        "/model.22/Add_1_output_0-rs",
        "/model.22/Sub_output_0-rs",
        "/model.22/Slice_output_0-rs",
        "/model.22/Slice_1_output_0-rs",
    ]

    fp16_keys = pick_existing_keys_in_cfg(cfg_path, candidates)

    if not fp16_keys:
        print("[WARN] No fp16 candidate keys found in cfg. (unexpected)")
    return fp16_keys


# =========================
# MAIN
# =========================
def main():
    print("I rknn-toolkit2 version will print below automatically")

    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}")

    if not DATASET_TXT.exists():
        raise FileNotFoundError(f"dataset.txt not found: {DATASET_TXT}")

    # reset work dir
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # step1은 dataset 경로를 WORK_DIR 내부에서 찾는 경우가 있어 복사(너가 이미 맞게 하고 있음)
    shutil.copy2(DATASET_TXT, WORK_DIR / "dataset.txt")

    # step1/step2 결과 파일을 WORK_DIR에 모으기 위해 chdir
    cwd = Path.cwd()
    os.chdir(WORK_DIR)

    try:
        rknn = RKNN(verbose=True)

        print("== RKNN config ==")
        rknn.config(**RKNN_CONFIG)

        print("== load onnx ==")
        ret = rknn.load_onnx(model=str((cwd / ONNX_PATH).resolve()))
        if ret != 0:
            raise RuntimeError(f"load_onnx failed ret={ret}")

        # -------------------------
        # STEP1
        # -------------------------
        print("== step1 (hybrid_quantization_step1) ==")
        ret = rknn.hybrid_quantization_step1(dataset="dataset.txt")
        if ret != 0:
            raise RuntimeError(f"step1 failed ret={ret}")

        # step1 outputs 찾기
        cfg_files = list(Path(".").glob("*.quantization.cfg"))
        model_files = list(Path(".").glob("*.model"))
        data_files = list(Path(".").glob("*.data"))

        if not cfg_files or not model_files or not data_files:
            raise RuntimeError(
                f"STEP1 outputs missing in {WORK_DIR}\n"
                f"cfg={len(cfg_files)}, model={len(model_files)}, data={len(data_files)}\n"
                f"Check step1 logs."
            )

        cfg_path = cfg_files[0]
        model_path = model_files[0]
        data_path = data_files[0]

        print(f"[STEP1] cfg  : {cfg_path}")
        print(f"[STEP1] model: {model_path}")
        print(f"[STEP1] data : {data_path}")

        # -------------------------
        # PATCH CFG (TEXT ONLY)
        # -------------------------
        print("== patch cfg (MODE=boxdfl+scorefix) ==")
        fp16_keys = build_fp16_keys_boxdfl_scorefix(cfg_path)
        print(f"== patch cfg: fp16_keys={len(fp16_keys)} ==")
        for k in fp16_keys:
            print("  fp16:", k)

        patch_cfg_text_only(cfg_path, fp16_keys)

        # -------------------------
        # STEP2
        # -------------------------
        print("== step2 (hybrid_quantization_step2) ==")
        ret = rknn.hybrid_quantization_step2(
            model_input=str(model_path),
            data_input=str(data_path),
            model_quantization_cfg=str(cfg_path),
        )
        if ret != 0:
            raise RuntimeError(f"step2 failed ret={ret}")

        # -------------------------
        # EXPORT RKNN
        # -------------------------
        print("== export rknn ==")
        out_abs = (cwd / OUT_RKNN).resolve()
        ret = rknn.export_rknn(str(out_abs))
        if ret != 0:
            raise RuntimeError(f"export_rknn failed ret={ret}")

        print(f"[OK] saved: {out_abs}")

    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
