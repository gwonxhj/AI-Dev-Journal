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
#include "rknn_api.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define MODEL_INPUT_SIZE 640
#define CONF_THRESHOLD 0.2
#define NMS_THRESHOLD 0.45
#define IOU_THRESHOLD_MAP 0.5

struct Detection {
    int class_id;
    float confidence;
    Rect2f box;
};

struct GroundTruth {
    int class_id;
    Rect2f box;
};

double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

float calculate_iou(const Rect2f& box1, const Rect2f& box2) {
    float x1 = max(box1.x, box2.x);
    float y1 = max(box1.y, box2.y);
    float x2 = min(box1.x + box1.width, box2.x + box2.width);
    float y2 = min(box1.y + box1.height, box2.y + box2.height);

    if (x1 >= x2 || y1 >= y2) return 0.0f;

    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - intersection;

    return union_area > 0 ? intersection / union_area : 0.0f;
}

vector<Detection> nms(vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) return detections;

    sort(detections.begin(), detections.end(),
         [](const Detection& a, const Detection& b) {
             return a.confidence > b.confidence;
         });

    vector<Detection> result;
    vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (detections[i].class_id == detections[j].class_id) {
                if (calculate_iou(detections[i].box, detections[j].box) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return result;
}

struct LetterboxInfo {
    Mat image;   // uint8 NHWC (0..255)
    float scale;
    int pad_w;
    int pad_h;
};

// ✅ --rgb 지원: letterbox 내부에서 BGR->RGB 변환 선택
LetterboxInfo letterbox(const Mat& img_bgr, int target_size, bool use_rgb,
                        Scalar color = Scalar(114, 114, 114)) {
    Mat base = img_bgr;

    if (use_rgb) {
        // OpenCV imread는 기본 BGR이므로, RGB 입력을 원하는 모델이면 변환
        cvtColor(img_bgr, base, COLOR_BGR2RGB);
    }

    int h = base.rows;
    int w = base.cols;
    float scale = min((float)target_size / w, (float)target_size / h);

    int new_w = (int)round(w * scale);
    int new_h = (int)round(h * scale);

    Mat resized;
    resize(base, resized, Size(new_w, new_h));

    int top = (target_size - new_h) / 2;
    int bottom = target_size - new_h - top;
    int left = (target_size - new_w) / 2;
    int right = target_size - new_w - left;

    Mat padded;
    copyMakeBorder(resized, padded, top, bottom, left, right, BORDER_CONSTANT, color);

    LetterboxInfo info;
    info.image = padded;
    info.scale = scale;
    info.pad_w = left;
    info.pad_h = top;

    return info;
}

Rect2f scale_coords(const Rect2f& box, float scale, int pad_w, int pad_h, int orig_w, int orig_h) {
    float x = (box.x - pad_w) / scale;
    float y = (box.y - pad_h) / scale;
    float w = box.width / scale;
    float h = box.height / scale;

    x = max(0.0f, min(x, (float)orig_w));
    y = max(0.0f, min(y, (float)orig_h));
    w = max(0.0f, min(w, (float)orig_w - x));
    h = max(0.0f, min(h, (float)orig_h - y));

    return Rect2f(x, y, w, h);
}

vector<Detection> postprocess_yolov8(rknn_output* outputs, int img_w, int img_h,
                                    float scale, int pad_w, int pad_h,
                                    float conf_thres, int num_classes) {
    vector<Detection> detections;
    float* data = (float*)outputs[0].buf;
    int num_anchors = 8400;

    for (int i = 0; i < num_anchors; i++) {
        float max_score = -1.0f;
        int class_id = -1;

        for (int c = 0; c < num_classes; c++) {
            float score = data[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score < conf_thres) continue;

        float cx = data[0 * num_anchors + i];
        float cy = data[1 * num_anchors + i];
        float w  = data[2 * num_anchors + i];
        float h  = data[3 * num_anchors + i];

        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;

        Rect2f box_model(x1, y1, w, h);
        Rect2f box_origin = scale_coords(box_model, scale, pad_w, pad_h, img_w, img_h);

        Detection det;
        det.class_id = class_id;
        det.confidence = max_score;
        det.box = box_origin;
        detections.push_back(det);
    }
    return detections;
}

static float calculate_map_at_iou(const vector<vector<Detection>>& all_predictions,
                                  const vector<vector<GroundTruth>>& all_ground_truths,
                                  int num_classes,
                                  float iou_thr)
{
    vector<float> aps(num_classes, 0.0f);
    int valid_classes = 0;

    for (int c = 0; c < num_classes; c++) {
        vector<pair<float, bool>> predictions; // (conf, is_tp)
        int total_gt = 0;

        for (size_t img_idx = 0; img_idx < all_predictions.size(); img_idx++) {
            vector<bool> gt_matched(all_ground_truths[img_idx].size(), false);

            for (const auto& pred : all_predictions[img_idx]) {
                if (pred.class_id != c) continue;

                float max_iou = 0.0f;
                int max_idx = -1;

                for (size_t j = 0; j < all_ground_truths[img_idx].size(); j++) {
                    if (all_ground_truths[img_idx][j].class_id != c || gt_matched[j]) continue;
                    float v = calculate_iou(pred.box, all_ground_truths[img_idx][j].box);
                    if (v > max_iou) { max_iou = v; max_idx = (int)j; }
                }

                bool is_tp = (max_iou >= iou_thr && max_idx >= 0);
                if (is_tp) gt_matched[max_idx] = true;

                predictions.push_back({pred.confidence, is_tp});
            }

            for (const auto& gt : all_ground_truths[img_idx]) {
                if (gt.class_id == c) total_gt++;
            }
        }

        if (total_gt == 0) continue;

        sort(predictions.begin(), predictions.end(),
             [](const pair<float, bool>& a, const pair<float, bool>& b) { return a.first > b.first; });

        vector<float> precisions, recalls;
        int tp = 0, fp = 0;

        for (const auto& p : predictions) {
            if (p.second) tp++;
            else fp++;
            precisions.push_back((float)tp / (tp + fp));
            recalls.push_back((float)tp / total_gt);
        }

        // 11-point AP(0.0~1.0 step 0.1)
        float ap = 0.0f;
        for (float t = 0.0f; t <= 1.0f; t += 0.1f) {
            float max_prec = 0.0f;
            for (size_t i = 0; i < recalls.size(); i++) {
                if (recalls[i] >= t) max_prec = max(max_prec, precisions[i]);
            }
            ap += max_prec;
        }
        ap /= 11.0f;

        aps[c] = ap;
        if (ap > 0) valid_classes++;
    }

    float sum = 0.0f;
    for (float ap : aps) sum += ap;
    return valid_classes > 0 ? sum / valid_classes : 0.0f;
}

static float calculate_map_coco_50(const vector<vector<Detection>>& all_predictions,
                                   const vector<vector<GroundTruth>>& all_ground_truths,
                                   int num_classes)
{
    return calculate_map_at_iou(all_predictions, all_ground_truths, num_classes, 0.50f);
}

static float calculate_map_coco_50_95(const vector<vector<Detection>>& all_predictions,
                                      const vector<vector<GroundTruth>>& all_ground_truths,
                                      int num_classes)
{
    float sum = 0.0f;
    int cnt = 0;

    for (float thr = 0.50f; thr <= 0.95f + 1e-6f; thr += 0.05f) {
        sum += calculate_map_at_iou(all_predictions, all_ground_truths, num_classes, thr);
        cnt++;
    }
    return (cnt > 0) ? (sum / cnt) : 0.0f;
}

void calculate_f1_score(const vector<vector<Detection>>& all_predictions,
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
                    float iou = calculate_iou(pred.box, all_ground_truths[i][j].box);
                    if (iou >= IOU_THRESHOLD_MAP) {
                        tp++;
                        gt_matched[j] = true;
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) fp++;
        }

        for (size_t j = 0; j < all_ground_truths[i].size(); j++) {
            if (!gt_matched[j]) fn++;
        }
    }

    precision = (tp + fp) > 0 ? (float)tp / (tp + fp) : 0.0f;
    recall    = (tp + fn) > 0 ? (float)tp / (tp + fn) : 0.0f;
    f1        = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0f;
}

vector<GroundTruth> load_ground_truth(const string& label_path, int img_w, int img_h) {
    vector<GroundTruth> gts;
    ifstream file(label_path);
    if (!file.is_open()) return gts;

    int class_id;
    float x, y, w, h;
    while (file >> class_id >> x >> y >> w >> h) {
        GroundTruth gt;
        gt.class_id = class_id;
        gt.box = Rect2f((x - w/2) * img_w, (y - h/2) * img_h, w * img_w, h * img_h);
        gts.push_back(gt);
    }
    return gts;
}

vector<string> get_image_files(const string& dir_path) {
    vector<string> files;
    DIR* dir = opendir(dir_path.c_str());
    if (dir == NULL) return files;

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

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <model.rknn> <image_dir> <label_dir> [num_classes] [--rgb]\n", argv[0]);
        printf("Example: %s /home/odroid/rise/yolov8m.rknn /home/odroid/rise/images /home/odroid/rise/labels 1 --rgb\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    const char* image_dir  = argv[2];
    const char* label_dir  = argv[3];
    int num_classes = (argc > 4) ? atoi(argv[4]) : 1;

    // ✅ --rgb 옵션 파싱
    bool use_rgb = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rgb") == 0) use_rgb = true;
    }

    // 헤더 출력(기존 + rgb 표시만 추가)
    printf("========================================\n");
    printf("     YOLOv8 RKNN Benchmark Tool\n");
    printf("========================================\n");
    printf("Model:          %s\n", model_path);
    printf("Images:         %s\n", image_dir);
    printf("Labels:         %s\n", label_dir);
    printf("Classes:        %d\n", num_classes);
    printf("Conf Threshold: %.2f\n", CONF_THRESHOLD);
    printf("NMS Threshold:  %.2f\n", NMS_THRESHOLD);
    printf("RGB Input:      %s\n", use_rgb ? "ON (--rgb)" : "OFF (BGR)");
    printf("========================================\n\n");

    // 모델 로드
    FILE* fp = fopen(model_path, "rb");
    if (fp == NULL) {
        printf("Failed to open model file: %s\n", model_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void* model_data = malloc(model_len);
    fread(model_data, 1, model_len, fp);
    fclose(fp);

    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_len, 0, NULL);
    free(model_data);
    if (ret < 0) {
        printf("rknn_init failed! ret=%d\n", ret);
        return -1;
    }

    // RK3588 3-core
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    rknn_set_core_mask(ctx, core_mask);

    printf("✓ Model loaded successfully\n");

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("✓ Model has %d inputs and %d outputs\n", io_num.n_input, io_num.n_output);

    // (참고) input/output attrs 출력 유지
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        printf("  Input[%d]: dims=%d %d %d %d, type=%d\n", i,
               input_attrs[i].dims[0], input_attrs[i].dims[1],
               input_attrs[i].dims[2], input_attrs[i].dims[3],
               input_attrs[i].type);
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        printf("  Output[%d]: dims=%d %d %d %d, type=%d\n", i,
               output_attrs[i].dims[0], output_attrs[i].dims[1],
               output_attrs[i].dims[2], output_attrs[i].dims[3],
               output_attrs[i].type);
    }

    vector<string> image_files = get_image_files(image_dir);
    if (image_files.empty()) {
        printf("No images found in %s\n", image_dir);
        rknn_destroy(ctx);
        return -1;
    }
    printf("✓ Found %zu images\n\n", image_files.size());

    vector<vector<Detection>> all_predictions;
    vector<vector<GroundTruth>> all_ground_truths;
    double total_time = 0;
    int frame_count = 0;

    printf("Processing images...\n");

    for (const auto& img_path : image_files) {
        Mat img = imread(img_path);
        if (img.empty()) {
            printf("Warning: Failed to load %s\n", img_path.c_str());
            continue;
        }

        int orig_w = img.cols;
        int orig_h = img.rows;

        // ✅ --rgb 반영된 letterbox
        LetterboxInfo lb_info = letterbox(img, MODEL_INPUT_SIZE, use_rgb);

        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type  = RKNN_TENSOR_UINT8;
        inputs[0].size  = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3;
        inputs[0].fmt   = RKNN_TENSOR_NHWC;
        inputs[0].buf   = lb_info.image.data;

        rknn_inputs_set(ctx, io_num.n_input, inputs);

        rknn_output outputs[io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < io_num.n_output; i++) outputs[i].want_float = 1;

        double start = get_current_time();
        rknn_run(ctx, NULL);
        rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        double end = get_current_time();

        total_time += (end - start);
        frame_count++;

        vector<Detection> dets = postprocess_yolov8(outputs, orig_w, orig_h,
                                                    lb_info.scale, lb_info.pad_w, lb_info.pad_h,
                                                    CONF_THRESHOLD, num_classes);
        dets = nms(dets, NMS_THRESHOLD);
        all_predictions.push_back(dets);

        string filename = img_path.substr(img_path.find_last_of("/") + 1);
        string label_path = string(label_dir) + "/" + filename.substr(0, filename.find_last_of(".")) + ".txt";
        all_ground_truths.push_back(load_ground_truth(label_path, orig_w, orig_h));

        rknn_outputs_release(ctx, io_num.n_output, outputs);

        if (frame_count % 10 == 0 || frame_count == 1) {
            printf("\rProcessed: %d/%zu images (Detections: %zu)",
                   frame_count, image_files.size(), dets.size());
            fflush(stdout);
        }
    }
    printf("\n\n");

    if (frame_count == 0) {
        printf("No images were processed successfully!\n");
        rknn_destroy(ctx);
        return -1;
    }

    float fps = frame_count / (total_time / 1000.0);
    float avg_inference_time = (float)(total_time / frame_count);

    float precision, recall, f1;
    calculate_f1_score(all_predictions, all_ground_truths, precision, recall, f1);

    float map50   = calculate_map_coco_50(all_predictions, all_ground_truths, num_classes);
    float map5095 = calculate_map_coco_50_95(all_predictions, all_ground_truths, num_classes);

    printf("========================================\n");
    printf("         BENCHMARK RESULTS\n");
    printf("========================================\n");
    printf("Total Images:        %d\n", frame_count);
    printf("Total Time:          %.2f ms\n", total_time);
    printf("Avg Inference Time:  %.2f ms\n", avg_inference_time);
    printf("FPS:                 %.2f\n", fps);
    printf("----------------------------------------\n");
    printf("Precision:           %.4f (%.2f%%)\n", precision, precision * 100);
    printf("Recall:              %.4f (%.2f%%)\n", recall, recall * 100);
    printf("F1 Score:            %.4f (%.2f%%)\n", f1, f1 * 100);
    printf("COCO mAP@50:         %.4f\n", map50);
    printf("COCO mAP@50-95:      %.4f\n", map5095);
    printf("========================================\n");

    rknn_destroy(ctx);
    return 0;
}
