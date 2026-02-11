#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <limits>

#include "rknn_api.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define MODEL_INPUT_SIZE 640
#define CONF_THRESHOLD 0.2f
#define NMS_THRESHOLD 0.45f
#define IOU_THRESHOLD_MAP 0.5f

struct Detection {
    int class_id;
    float confidence;
    Rect2f box;
};

struct GroundTruth {
    int class_id;
    Rect2f box;
};

struct LetterboxInfo {
    Mat img_u8;   // uint8 NHWC
    Mat img_f32;  // float32 NHWC 0..1 (참고용)
    float scale;
    int pad_w, pad_h;
};

static inline double now_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static inline float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

static inline float calculate_iou(const Rect2f& a, const Rect2f& b) {
    float x1 = max(a.x, b.x);
    float y1 = max(a.y, b.y);
    float x2 = min(a.x + a.width,  b.x + b.width);
    float y2 = min(a.y + a.height, b.y + b.height);
    if (x1 >= x2 || y1 >= y2) return 0.f;
    float inter = (x2 - x1) * (y2 - y1);
    float uni   = a.width * a.height + b.width * b.height - inter;
    return inter / uni;
}

static vector<Detection> nms(vector<Detection>& dets, float thresh) {
    sort(dets.begin(), dets.end(),
         [](const Detection& a, const Detection& b){ return a.confidence > b.confidence; });

    vector<Detection> res;
    vector<bool> removed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); i++) {
        if (removed[i]) continue;
        res.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (removed[j]) continue;
            if (dets[i].class_id == dets[j].class_id &&
                calculate_iou(dets[i].box, dets[j].box) > thresh) {
                removed[j] = true;
            }
        }
    }
    return res;
}

// letterbox: NHWC 유지, 기본 BGR, --rgb면 RGB
static LetterboxInfo letterbox(const Mat& bgr, bool rgb) {
    Mat base = bgr;
    Mat converted;
    if (rgb) {
        cvtColor(bgr, converted, COLOR_BGR2RGB);
        base = converted;
    }

    float scale = min((float)MODEL_INPUT_SIZE / base.cols,
                      (float)MODEL_INPUT_SIZE / base.rows);

    int nw = (int)round(base.cols * scale);
    int nh = (int)round(base.rows * scale);

    Mat resized;
    resize(base, resized, Size(nw, nh), 0, 0, INTER_LINEAR);

    int pw = (MODEL_INPUT_SIZE - nw) / 2;
    int ph = (MODEL_INPUT_SIZE - nh) / 2;

    Mat out_u8;
    copyMakeBorder(resized, out_u8,
                   ph, MODEL_INPUT_SIZE - nh - ph,
                   pw, MODEL_INPUT_SIZE - nw - pw,
                   BORDER_CONSTANT, Scalar(114,114,114));

    Mat out_f32;
    out_u8.convertTo(out_f32, CV_32FC3, 1.0/255.0);

    return {out_u8, out_f32, scale, pw, ph};
}

static inline Rect2f scale_coords_xywh(
    float x1, float y1, float w, float h,
    float scale, int pad_w, int pad_h,
    int orig_w, int orig_h)
{
    float x = (x1 - pad_w) / scale;
    float y = (y1 - pad_h) / scale;
    float ww = w / scale;
    float hh = h / scale;

    x = max(0.f, min(x, (float)orig_w - 1));
    y = max(0.f, min(y, (float)orig_h - 1));

    ww = max(0.f, min(ww, (float)orig_w - x));
    hh = max(0.f, min(hh, (float)orig_h - y));

    return Rect2f(x, y, ww, hh);
}

// split output postprocess: boxes/scores 모두 float로 받은 뒤 처리
static vector<Detection> postprocess_yolov8_split(
    const float* boxes, const rknn_tensor_attr& attr_boxes,
    const float* scores, const rknn_tensor_attr& attr_scores,
    int orig_w, int orig_h,
    float lb_scale, int pad_w, int pad_h)
{
    vector<Detection> dets;

    if (attr_boxes.n_dims < 3 || attr_scores.n_dims < 3) {
        printf("ERROR: outputs must be 3D.\n");
        return dets;
    }

    int b1 = attr_boxes.dims[1], b2 = attr_boxes.dims[2];
    int s1 = attr_scores.dims[1], s2 = attr_scores.dims[2];

    bool boxes_ch_first = false;
    int N = -1;
    if (b1 == 4) { boxes_ch_first = true; N = b2; }
    else if (b2 == 4) { boxes_ch_first = false; N = b1; }
    else {
        printf("ERROR: cannot infer boxes layout from dims [1,%d,%d]\n", b1, b2);
        return dets;
    }

    bool scores_ch_first = false;
    int Ns = -1;
    if (s1 == 1) { scores_ch_first = true; Ns = s2; }
    else if (s2 == 1) { scores_ch_first = false; Ns = s1; }
    else {
        printf("ERROR: cannot infer scores layout from dims [1,%d,%d]\n", s1, s2);
        return dets;
    }

    if (Ns != N) {
        printf("ERROR: N mismatch boxes=%d scores=%d\n", N, Ns);
        return dets;
    }

    auto get_box = [&](int i, float& cx, float& cy, float& w, float& h) {
        if (boxes_ch_first) {
            cx = boxes[0 * N + i];
            cy = boxes[1 * N + i];
            w  = boxes[2 * N + i];
            h  = boxes[3 * N + i];
        } else {
            cx = boxes[i * 4 + 0];
            cy = boxes[i * 4 + 1];
            w  = boxes[i * 4 + 2];
            h  = boxes[i * 4 + 3];
        }
    };

    // C=1 이라서 사실상 어떤 레이아웃이든 scores[i]가 맞음
    auto get_score = [&](int i) -> float {
        (void)scores_ch_first;
        return scores[i];
    };

    static bool printed_once = false;
    if (!printed_once) {
        printf("POST SPLIT OK: boxes=%s scores=%s N=%d\n",
               boxes_ch_first ? "[1,4,N]" : "[1,N,4]",
               scores_ch_first ? "[1,1,N]" : "[1,N,1]",
               N);

        float mn = +std::numeric_limits<float>::infinity();
        float mx = -std::numeric_limits<float>::infinity();
        int sampleN = std::min(N, 4096);
        for (int i = 0; i < sampleN; i++) {
            float v = get_score(i);
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        printf("[SCORE sample] min=%.6f max=%.6f (sampleN=%d)\n", mn, mx, sampleN);
        printed_once = true;
    }

    static bool printed_raw_for_first = false;
    int raw_left = printed_raw_for_first ? 0 : 10;

    for (int i = 0; i < N; i++) {
        float raw = get_score(i);
        float sc = (raw >= -0.001f && raw <= 1.001f) ? raw : sigmoid(raw);
        if (sc < CONF_THRESHOLD) continue;

        float cx, cy, w, h;
        get_box(i, cx, cy, w, h);

        // 정규화 좌표 방어
        if (cx <= 2.f && cy <= 2.f && w <= 2.f && h <= 2.f) {
            cx *= MODEL_INPUT_SIZE;
            cy *= MODEL_INPUT_SIZE;
            w  *= MODEL_INPUT_SIZE;
            h  *= MODEL_INPUT_SIZE;
        }

        if (raw_left > 0) {
            printf("[RAW] i=%d cx=%.2f cy=%.2f w=%.2f h=%.2f score=%.3f\n", i, cx, cy, w, h, sc);
            raw_left--;
            if (raw_left == 0) printed_raw_for_first = true;
        }

        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;

        Rect2f box_origin = scale_coords_xywh(x1, y1, w, h, lb_scale, pad_w, pad_h, orig_w, orig_h);
        if (box_origin.width <= 1.f || box_origin.height <= 1.f) continue;

        dets.push_back({0, sc, box_origin});
    }

    return dets;
}

// ===================== METRICS: COCO-style mAP =====================

// COCO AP at a single IoU threshold using 101-point interpolation + precision envelope
static float calculate_coco_ap_iou(
    const vector<vector<Detection>>& all_predictions,
    const vector<vector<GroundTruth>>& all_ground_truths,
    int num_classes,
    float iou_thresh)
{
    vector<float> aps(num_classes, 0.0f);
    int valid_classes = 0;

    for (int c = 0; c < num_classes; c++) {
        // For COCO AP, we aggregate predictions across images per class
        vector<pair<float, bool>> preds; // (conf, is_tp)
        int total_gt = 0;

        for (size_t img_idx = 0; img_idx < all_predictions.size(); img_idx++) {
            const auto& gts = all_ground_truths[img_idx];
            vector<bool> gt_matched(gts.size(), false);

            // count GTs of this class
            for (const auto& gt : gts) if (gt.class_id == c) total_gt++;

            // match predictions (greedy per image, by confidence order is achieved globally later)
            // here we just compute TP/FP labels for each pred against best unmatched GT
            for (const auto& pred : all_predictions[img_idx]) {
                if (pred.class_id != c) continue;

                float best_iou = 0.0f;
                int best_j = -1;

                for (size_t j = 0; j < gts.size(); j++) {
                    if (gt_matched[j]) continue;
                    if (gts[j].class_id != c) continue;

                    float v = calculate_iou(pred.box, gts[j].box);
                    if (v > best_iou) { best_iou = v; best_j = (int)j; }
                }

                bool is_tp = (best_iou >= iou_thresh && best_j >= 0);
                if (is_tp) gt_matched[best_j] = true;

                preds.push_back({pred.confidence, is_tp});
            }
        }

        if (total_gt == 0) continue;

        sort(preds.begin(), preds.end(),
             [](const pair<float, bool>& a, const pair<float, bool>& b){
                 return a.first > b.first;
             });

        // Build PR curve
        int tp = 0, fp = 0;
        vector<float> precisions;
        vector<float> recalls;
        precisions.reserve(preds.size());
        recalls.reserve(preds.size());

        for (const auto& p : preds) {
            if (p.second) tp++;
            else fp++;

            float prec = (tp + fp) ? (float)tp / (tp + fp) : 0.0f;
            float rec  = (float)tp / (float)total_gt;

            precisions.push_back(prec);
            recalls.push_back(rec);
        }

        if (precisions.empty()) {
            aps[c] = 0.0f;
            continue;
        }

        // Precision envelope: make precision non-increasing w.r.t. recall (from end to start)
        for (int i = (int)precisions.size() - 2; i >= 0; --i) {
            if (precisions[i] < precisions[i + 1]) precisions[i] = precisions[i + 1];
        }

        // 101-point interpolation
        float ap = 0.0f;
        for (int ri = 0; ri <= 100; ri++) {
            float r = ri / 100.0f; // 0.00 .. 1.00
            float pmax = 0.0f;

            // find first index where recall >= r, then take precision at that index (enveloped)
            // COCO style: p_interp(r) = max_{r' >= r} p(r') which envelope already ensures,
            // so picking first recall>=r is enough after envelope.
            for (size_t i = 0; i < recalls.size(); i++) {
                if (recalls[i] >= r) { pmax = precisions[i]; break; }
            }

            ap += pmax;
        }
        ap /= 101.0f;

        aps[c] = ap;
        if (ap > 0.0f) valid_classes++;
    }

    float sum = 0.0f;
    for (float ap : aps) sum += ap;

    return valid_classes > 0 ? sum / valid_classes : 0.0f;
}

// COCO mAP@50 (IoU=0.50)
static float calculate_coco_map50(
    const vector<vector<Detection>>& all_predictions,
    const vector<vector<GroundTruth>>& all_ground_truths,
    int num_classes)
{
    return calculate_coco_ap_iou(all_predictions, all_ground_truths, num_classes, 0.50f);
}

// COCO mAP@50:95 (IoU=0.50..0.95 step 0.05)
static float calculate_coco_map50_95(
    const vector<vector<Detection>>& all_predictions,
    const vector<vector<GroundTruth>>& all_ground_truths,
    int num_classes)
{
    float sum = 0.0f;
    for (int k = 0; k < 10; k++) {
        float thr = 0.50f + 0.05f * k;
        sum += calculate_coco_ap_iou(all_predictions, all_ground_truths, num_classes, thr);
    }
    return sum / 10.0f;
}

static void calculate_f1_score(const vector<vector<Detection>>& all_predictions,
                               const vector<vector<GroundTruth>>& all_ground_truths,
                               float& precision, float& recall, float& f1) {
    int tp = 0, fp = 0, fn = 0;

    for (size_t i = 0; i < all_predictions.size(); i++) {
        vector<bool> gt_matched(all_ground_truths[i].size(), false);

        for (const auto& pred : all_predictions[i]) {
            bool matched = false;
            for (size_t j = 0; j < all_ground_truths[i].size(); j++) {
                if (gt_matched[j]) continue;
                if (pred.class_id == all_ground_truths[i][j].class_id) {
                    float v = calculate_iou(pred.box, all_ground_truths[i][j].box);
                    if (v >= IOU_THRESHOLD_MAP) {
                        tp++;
                        gt_matched[j] = true;
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) fp++;
        }

        for (size_t j = 0; j < all_ground_truths[i].size(); j++) if (!gt_matched[j]) fn++;
    }

    precision = (tp + fp) ? (float)tp / (tp + fp) : 0.0f;
    recall    = (tp + fn) ? (float)tp / (tp + fn) : 0.0f;
    f1        = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0.0f;
}

static vector<GroundTruth> load_ground_truth(const string& label_path, int img_w, int img_h) {
    vector<GroundTruth> gts;
    ifstream file(label_path);
    if (!file.is_open()) return gts;

    int class_id;
    float x, y, w, h;
    while (file >> class_id >> x >> y >> w >> h) {
        gts.push_back({class_id, Rect2f((x - w/2) * img_w, (y - h/2) * img_h, w * img_w, h * img_h)});
    }
    file.close();
    return gts;
}

static vector<string> get_image_files(const string& dir_path) {
    vector<string> files;
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) return files;

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        string filename = entry->d_name;
        if (filename.find(".jpg") != string::npos ||
            filename.find(".png") != string::npos ||
            filename.find(".jpeg") != string::npos) {
            files.push_back(dir_path + "/" + filename);
        }
    }
    closedir(dir);
    sort(files.begin(), files.end());
    return files;
}

// attr에서 element 개수 계산
static size_t numel_from_attr(const rknn_tensor_attr& a) {
    size_t n = 1;
    for (int i = 0; i < a.n_dims; i++) n *= (size_t)a.dims[i];
    return n;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <model.rknn> <image_dir> <label_dir> [num_classes] [--rgb] [--force_float] [--dump_raw_score]\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    const char* image_dir  = argv[2];
    const char* label_dir  = argv[3];
    int num_classes = (argc > 4) ? atoi(argv[4]) : 1;

    bool use_rgb = false;
    bool force_float = false;
    bool dump_raw_score = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rgb") == 0) use_rgb = true;
        if (strcmp(argv[i], "--force_float") == 0) force_float = true;
        if (strcmp(argv[i], "--dump_raw_score") == 0) dump_raw_score = true;
    }

    FILE* fp = fopen(model_path, "rb");
    if (!fp) { printf("Failed to open model file: %s\n", model_path); return -1; }

    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    void* model_data = malloc(model_len);
    size_t rs = fread(model_data, 1, model_len, fp);
    fclose(fp);
    if (rs != (size_t)model_len) {
        printf("ERROR: model fread failed\n");
        free(model_data);
        return -1;
    }

    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_len, 0, NULL);
    free(model_data);
    if (ret < 0) { printf("rknn_init failed! ret=%d\n", ret); return -1; }

    ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
    if (ret != RKNN_SUCC) printf("WARNING: rknn_set_core_mask failed ret=%d\n", ret);

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("RKNN IO: inputs=%d outputs=%d\n", io_num.n_input, io_num.n_output);

    // Input attr
    rknn_tensor_attr in_attr{};
    in_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    printf("IN[0]: n_dims=%d dims=", in_attr.n_dims);
    for (int k = 0; k < in_attr.n_dims; k++) printf("%d ", in_attr.dims[k]);
    printf(" type=%d fmt=%d qnt_type=%d zp=%d scale=%f\n",
           in_attr.type, in_attr.fmt, in_attr.qnt_type, in_attr.zp, in_attr.scale);

    bool model_input_is_u8  = (in_attr.type == RKNN_TENSOR_UINT8);
    bool model_input_is_f32 = (in_attr.type == RKNN_TENSOR_FLOAT32);

    // 안전장치: UINT8 입력 모델인데 float 강제면 거부
    if (force_float && model_input_is_u8) {
        printf("[WARN] --force_float requested, but model input type is UINT8.\n");
        printf("       This model is compiled for UINT8 input (qnt_type=%d zp=%d scale=%f).\n", in_attr.qnt_type, in_attr.zp, in_attr.scale);
        printf("       Forcing float input may crash. Ignoring --force_float.\n");
        force_float = false;
    }

    // Output attrs + boxes/scores index
    vector<rknn_tensor_attr> out_attrs(io_num.n_output);
    int idx_boxes = -1, idx_scores = -1;

    for (int i = 0; i < io_num.n_output; i++) {
        rknn_tensor_attr a{};
        a.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
        out_attrs[i] = a;

        printf("OUT[%d]: n_dims=%d dims=", i, a.n_dims);
        for (int k = 0; k < a.n_dims; k++) printf("%d ", a.dims[k]);
        printf(" type=%d fmt=%d qnt_type=%d zp=%d scale=%f\n",
               a.type, a.fmt, a.qnt_type, a.zp, a.scale);

        if (a.n_dims >= 3) {
            if (a.dims[1] == 4 || a.dims[2] == 4) idx_boxes = i;
            if (a.dims[1] == 1 || a.dims[2] == 1) idx_scores = i;
        }
    }

    if (idx_boxes < 0 || idx_scores < 0) {
        printf("ERROR: split outputs not found (need boxes C=4, scores C=1)\n");
        rknn_destroy(ctx);
        return -1;
    }

    vector<string> image_files = get_image_files(image_dir);
    if (image_files.empty()) {
        printf("No images found in %s\n", image_dir);
        rknn_destroy(ctx);
        return -1;
    }

    // =========================
    // ✅ 임시버퍼(프리얼록) 준비
    // =========================
    size_t boxes_numel  = numel_from_attr(out_attrs[idx_boxes]);
    size_t scores_numel = numel_from_attr(out_attrs[idx_scores]);

    // want_float=1로 받을 때의 타입은 float32
    vector<float> boxes_f32(boxes_numel);
    vector<float> scores_f32(scores_numel);

    // raw score dump용 (want_float=0) -> int8로 가정
    // (실제로 scores 출력이 float인 모델이면 raw는 의미 없지만, dump를 요청했으니 안전하게 준비)
    vector<int8_t> scores_i8(scores_numel);

    // prealloc outputs 배열을 2세트 준비: (float용 / raw용)
    vector<rknn_output> outs_float(io_num.n_output);
    vector<rknn_output> outs_raw(io_num.n_output);

    for (int i = 0; i < io_num.n_output; i++) {
        memset(&outs_float[i], 0, sizeof(rknn_output));
        outs_float[i].index = i;
        outs_float[i].want_float = 1;
        outs_float[i].is_prealloc = 1;

        memset(&outs_raw[i], 0, sizeof(rknn_output));
        outs_raw[i].index = i;
        outs_raw[i].want_float = 1;
        outs_raw[i].is_prealloc = 1;
    }

    // float 출력 버퍼 바인딩
    outs_float[idx_boxes].buf  = (void*)boxes_f32.data();
    outs_float[idx_boxes].size = boxes_f32.size() * sizeof(float);

    outs_float[idx_scores].buf  = (void*)scores_f32.data();
    outs_float[idx_scores].size = scores_f32.size() * sizeof(float);

    // raw 출력 버퍼 바인딩 (scores만 raw)
    outs_raw[idx_boxes].want_float = 1; // boxes는 float로 유지
    outs_raw[idx_boxes].buf  = (void*)boxes_f32.data();
    outs_raw[idx_boxes].size = boxes_f32.size() * sizeof(float);

    outs_raw[idx_scores].want_float = 0; // scores raw
    outs_raw[idx_scores].buf  = (void*)scores_i8.data();
    outs_raw[idx_scores].size = scores_i8.size() * sizeof(int8_t);

    // =========================

    vector<vector<Detection>> all_predictions;
    vector<vector<GroundTruth>> all_ground_truths;

    double total_time = 0;
    int frame_count = 0;

    bool raw_dumped_once = false;

    for (const auto& img_path : image_files) {
        Mat img = imread(img_path);
        if (img.empty()) continue;

        int orig_w = img.cols;
        int orig_h = img.rows;

        LetterboxInfo lb = letterbox(img, use_rgb);

        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].fmt = RKNN_TENSOR_NHWC;

        if (force_float && model_input_is_f32) {
            inputs[0].type = RKNN_TENSOR_FLOAT32;
            inputs[0].size = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3 * sizeof(float);
            inputs[0].buf  = (void*)lb.img_f32.data;
            inputs[0].pass_through = 1;
        } else {
            // (대부분) UINT8 입력
            inputs[0].type = RKNN_TENSOR_UINT8;
            inputs[0].size = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3;
            inputs[0].buf  = (void*)lb.img_u8.data;
            inputs[0].pass_through = 0;
        }

        ret = rknn_inputs_set(ctx, 1, inputs);
        if (ret < 0) continue;

        // ---- (선택) raw score dump: 딱 1회, prealloc로 안전 수행 ----
        if (dump_raw_score && !raw_dumped_once) {
            raw_dumped_once = true;

            ret = rknn_run(ctx, NULL);
            if (ret == RKNN_SUCC && rknn_outputs_get(ctx, io_num.n_output, outs_raw.data(), NULL) == RKNN_SUCC) {
                int N = out_attrs[idx_scores].dims[2];
                int sampleN = std::min(N, 128);
                int mn = 127, mx = -128, cnt_m128 = 0;

                for (int i=0;i<sampleN;i++){
                    int v = (int)scores_i8[i];
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                    if (v == -128) cnt_m128++;
                }

                printf("[SCORE RAW int8 sample] min=%d max=%d (-128 count=%d/%d)\n",
                       mn, mx, cnt_m128, sampleN);

                // prealloc이면 release 불필요지만, 일부 런타임은 내부 상태 정리를 위해 호출해도 무해
                rknn_outputs_release(ctx, io_num.n_output, outs_raw.data());
            } else {
                printf("[WARN] raw score dump failed.\n");
            }
            // 이후 정상 플로우에서 다시 run 수행
        }

        double start = now_ms();
        ret = rknn_run(ctx, NULL);
        if (ret < 0) continue;

        ret = rknn_outputs_get(ctx, io_num.n_output, outs_float.data(), NULL);
        if (ret < 0) continue;
        double end = now_ms();

        total_time += (end - start);
        frame_count++;

        const float* boxes  = boxes_f32.data();
        const float* scores = scores_f32.data();

        auto dets = postprocess_yolov8_split(
            boxes, out_attrs[idx_boxes],
            scores, out_attrs[idx_scores],
            orig_w, orig_h,
            lb.scale, lb.pad_w, lb.pad_h
        );

        dets = nms(dets, NMS_THRESHOLD);
        all_predictions.push_back(dets);

        string filename = img_path.substr(img_path.find_last_of("/") + 1);
        string label_path = string(label_dir) + "/" + filename.substr(0, filename.find_last_of(".")) + ".txt";
        all_ground_truths.push_back(load_ground_truth(label_path, orig_w, orig_h));

        rknn_outputs_release(ctx, io_num.n_output, outs_float.data());
    }

    if (frame_count == 0) {
        printf("No images were processed successfully!\n");
        rknn_destroy(ctx);
        return -1;
    }

    float fps = frame_count / (total_time / 1000.0f);
    float avg_inference_time = (float)(total_time / frame_count);

    float precision, recall, f1;
    calculate_f1_score(all_predictions, all_ground_truths, precision, recall, f1);
    float map50 = calculate_coco_map50(all_predictions, all_ground_truths, num_classes);
    float map50_95 = calculate_coco_map50_95(all_predictions, all_ground_truths, num_classes);


    // ===== BENCHMARK RESULTS (요청대로 유지) =====
    printf("========================================\n");
    printf("         BENCHMARK RESULTS\n");
    printf("========================================\n");
    printf("Total Images:        %d\n", frame_count);
    printf("Total Time:          %.2f ms\n", total_time);
    printf("Avg Inference Time:  %.2f ms\n", avg_inference_time);
    printf("FPS:                 %.2f\n", fps);
    printf("----------------------------------------\n");
    printf("Precision:           %.4f\n", precision);
    printf("Recall:              %.4f\n", recall);
    printf("F1 Score:            %.4f\n", f1);
    printf("COCO mAP@50:         %.4f\n", map50);
    printf("COCO mAP@50-95:      %.4f\n", map50_95);
    printf("========================================\n");

    rknn_destroy(ctx);
    return 0;
}
