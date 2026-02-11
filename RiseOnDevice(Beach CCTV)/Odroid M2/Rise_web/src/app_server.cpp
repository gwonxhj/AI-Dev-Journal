#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include <condition_variable>

#include "rknn_api.h"
#include <opencv2/opencv.hpp>

#include "httplib.h"  // third_party/httplib.h

using namespace std;
using namespace cv;

static constexpr int   MODEL_INPUT_SIZE = 640;

static inline void sleep_ms(double ms) {
  if (ms <= 0) return;
  usleep((useconds_t)(ms * 1000.0));
}

struct Detection {
  int class_id;
  float confidence;
  Rect2f box;
};

static inline double now_ms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

static inline float iou(const Rect2f& a, const Rect2f& b) {
  float x1 = max(a.x, b.x);
  float y1 = max(a.y, b.y);
  float x2 = min(a.x + a.width,  b.x + b.width);
  float y2 = min(a.y + a.height, b.y + b.height);
  if (x1 >= x2 || y1 >= y2) return 0.f;
  float inter = (x2 - x1) * (y2 - y1);
  float uni = a.width * a.height + b.width * b.height - inter;
  return inter / uni;
}

static vector<Detection> nms(vector<Detection>& dets, float thr) {
  sort(dets.begin(), dets.end(),
       [](const Detection& a, const Detection& b){ return a.confidence > b.confidence; });
  vector<Detection> res;
  vector<bool> removed(dets.size(), false);
  for (size_t i = 0; i < dets.size(); i++) {
    if (removed[i]) continue;
    res.push_back(dets[i]);
    for (size_t j = i + 1; j < dets.size(); j++) {
      if (removed[j]) continue;
      if (dets[i].class_id == dets[j].class_id && iou(dets[i].box, dets[j].box) > thr) {
        removed[j] = true;
      }
    }
  }
  return res;
}

struct LetterboxInfo {
  Mat img_u8;   // uint8 NHWC 640x640
  float scale;
  int pad_w, pad_h;
};

static LetterboxInfo letterbox_u8(const Mat& bgr, bool rgb) {
  Mat base = bgr;
  Mat converted;
  if (rgb) { cvtColor(bgr, converted, COLOR_BGR2RGB); base = converted; }

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

  return {out_u8, scale, pw, ph};
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

// ---- FP16/FP32 single output: [1, C, N] (channel-first) assumed ----
// C = 4 + num_classes (여기선 class=1 고정이라 C=5 기대)
// scores가 이미 sigmoid된 형태면 그대로, logits면 sigmoid로 보정(휴리스틱)
static vector<Detection> postprocess_single_output(
  const float* data, int C, int N,
  int orig_w, int orig_h,
  float scale, int pad_w, int pad_h,
  float conf_thres, int num_classes)
{
  vector<Detection> dets;
  if (C < 5) return dets;

  for (int i = 0; i < N; i++) {
    float best = -1.f;
    int cid = 0;

    // person 1개면 사실상 c=0 하나지만, 일반화 유지
    for (int c = 0; c < num_classes; c++) {
      float v = data[(4 + c) * N + i];
      // logits 가능성 방어
      float sc = (v >= -0.001f && v <= 1.001f) ? v : sigmoid(v);
      if (sc > best) { best = sc; cid = c; }
    }
    if (best < conf_thres) continue;

    float cx = data[0 * N + i];
    float cy = data[1 * N + i];
    float w  = data[2 * N + i];
    float h  = data[3 * N + i];

    // 정규화 좌표 방어
    if (cx <= 2.f && cy <= 2.f && w <= 2.f && h <= 2.f) {
      cx *= MODEL_INPUT_SIZE; cy *= MODEL_INPUT_SIZE;
      w  *= MODEL_INPUT_SIZE; h  *= MODEL_INPUT_SIZE;
    }

    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;

    Rect2f box = scale_coords_xywh(x1, y1, w, h, scale, pad_w, pad_h, orig_w, orig_h);
    if (box.width <= 1.f || box.height <= 1.f) continue;

    dets.push_back({cid, best, box});
  }
  return dets;
}

// ---- Hybrid split output: boxes + scores (둘 다 want_float=1로 받음) ----
static vector<Detection> postprocess_split_output(
  const float* boxes, const rknn_tensor_attr& ab,
  const float* scores, const rknn_tensor_attr& as,
  int orig_w, int orig_h,
  float scale, int pad_w, int pad_h,
  float conf_thres)
{
  vector<Detection> dets;

  int b1 = ab.dims[1], b2 = ab.dims[2];
  bool boxes_ch_first = false;
  int N = -1;
  if (b1 == 4) { boxes_ch_first = true; N = b2; }
  else if (b2 == 4) { boxes_ch_first = false; N = b1; }
  else return dets;

  int s1 = as.dims[1], s2 = as.dims[2];
  int Ns = -1;
  if (s1 == 1) Ns = s2;
  else if (s2 == 1) Ns = s1;
  if (Ns != N) return dets;

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

  for (int i = 0; i < N; i++) {
    float raw = scores[i];
    float sc = (raw >= -0.001f && raw <= 1.001f) ? raw : sigmoid(raw);
    if (sc < conf_thres) continue;

    float cx, cy, w, h;
    get_box(i, cx, cy, w, h);

    if (cx <= 2.f && cy <= 2.f && w <= 2.f && h <= 2.f) {
      cx *= MODEL_INPUT_SIZE; cy *= MODEL_INPUT_SIZE;
      w  *= MODEL_INPUT_SIZE; h  *= MODEL_INPUT_SIZE;
    }

    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;

    Rect2f box = scale_coords_xywh(x1, y1, w, h, scale, pad_w, pad_h, orig_w, orig_h);
    if (box.width <= 1.f || box.height <= 1.f) continue;

    dets.push_back({0, sc, box});
  }

  return dets;
}

static bool read_file(const string& path, vector<uint8_t>& out) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) return false;
  fseek(fp, 0, SEEK_END);
  long len = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  out.resize(len);
  size_t rs = fread(out.data(), 1, len, fp);
  fclose(fp);
  return rs == (size_t)len;
}

struct AppConfig {
  string backend = "fp16"; // fp16 | int8
  string size = "s";      // n/s/m/l
  bool rgb = true;
  float conf = 0.20f;
  float nms = 0.45f;
  string video = "test.mp4";
  int jpeg_quality = 75;
  int stream_skip = 1;
};

class ModelRunner {
public:
  bool load(const string& model_path) {
    unload();

    vector<uint8_t> buf;
    if (!read_file(model_path, buf)) {
      cerr << "[ModelRunner] failed to read model: " << model_path << "\n";
      return false;
    }

    int ret = rknn_init(&ctx_, buf.data(), (int)buf.size(), 0, NULL);
    if (ret != RKNN_SUCC) {
      cerr << "[ModelRunner] rknn_init failed ret=" << ret << "\n";
      ctx_ = 0;
      return false;
    }

    rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0_1_2);

    rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));

    // input attr
    memset(&in_attr_, 0, sizeof(in_attr_));
    in_attr_.index = 0;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr_, sizeof(in_attr_));

    // output attrs
    out_attrs_.assign(io_num_.n_output, {});
    for (int i = 0; i < io_num_.n_output; i++) {
      rknn_tensor_attr a{};
      a.index = i;
      rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
      out_attrs_[i] = a;
    }

    // prepare prealloc buffers
    prepare_outputs();

    cout << "[ModelRunner] loaded: " << model_path
         << " outputs=" << io_num_.n_output
         << " in_type=" << in_attr_.type
         << " qnt_type=" << in_attr_.qnt_type
         << " zp=" << in_attr_.zp
         << " scale=" << in_attr_.scale
         << "\n";

    return true;
  }

  void unload() {
    if (ctx_) {
      rknn_destroy(ctx_);
      ctx_ = 0;
    }
    outs_.clear();
    out_buf_f32_.clear();
    idx_boxes_ = -1;
    idx_scores_ = -1;
  }

  bool ready() const { return ctx_ != 0; }

  // infer on a BGR frame, returns detections (class 1 = person)
  vector<Detection> infer(const Mat& frame_bgr, bool rgb, float conf, float nms_thr, double& infer_ms) {
    vector<Detection> dets;
    infer_ms = 0.0;

    if (!ctx_) return dets;

    int orig_w = frame_bgr.cols;
    int orig_h = frame_bgr.rows;

    LetterboxInfo lb = letterbox_u8(frame_bgr, rgb);

    rknn_input in{};
    in.index = 0;
    in.fmt   = RKNN_TENSOR_NHWC;
    in.type  = RKNN_TENSOR_UINT8;
    in.size  = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3;
    in.buf   = (void*)lb.img_u8.data;
    in.pass_through = 0;

    // 변환 설정이 uint8 입력(qnt_type=2 zp=-128 scale=1/255) 패턴이므로
    // 여기서는 "항상 u8 + pass_through=0"로 통일
    // float 입력 모델이 나오면 그때만 분기 추가

    if (rknn_inputs_set(ctx_, 1, &in) != RKNN_SUCC) return dets;

    double t0 = now_ms();
    if (rknn_run(ctx_, NULL) != RKNN_SUCC) return dets;

    if (rknn_outputs_get(ctx_, io_num_.n_output, outs_.data(), NULL) != RKNN_SUCC) return dets;
    double t1 = now_ms();
    infer_ms = (t1 - t0);

    if (io_num_.n_output == 1) {
      // single output float32
      // expect [1, C, N] or [1, N, C] but 대부분 [1,C,N]
      const auto& a = out_attrs_[0];
      int d1 = a.dims[1], d2 = a.dims[2];
      int C = d1, N = d2;
      // 방어: [1,N,C]면 swap
      if (d2 <= 512 && d1 > d2) { N = d1; C = d2; }

      const float* data = (const float*)outs_[0].buf;
      dets = postprocess_single_output(data, C, N, orig_w, orig_h, lb.scale, lb.pad_w, lb.pad_h, conf, 1);
    } else {
      // split outputs
      const float* boxes  = (const float*)outs_[idx_boxes_].buf;
      const float* scores = (const float*)outs_[idx_scores_].buf;
      dets = postprocess_split_output(
        boxes, out_attrs_[idx_boxes_],
        scores, out_attrs_[idx_scores_],
        orig_w, orig_h,
        lb.scale, lb.pad_w, lb.pad_h,
        conf
      );
    }

    dets = nms(dets, nms_thr);

    // prealloc outputs라도 release는 호출(런타임 내부 상태 안정 목적)
    rknn_outputs_release(ctx_, io_num_.n_output, outs_.data());

    return dets;
  }

private:
  void prepare_outputs() {
    outs_.assign(io_num_.n_output, {});
    out_buf_f32_.assign(io_num_.n_output, {});

    idx_boxes_ = -1;
    idx_scores_ = -1;

    // identify boxes/scores for split
    for (int i = 0; i < io_num_.n_output; i++) {
      const auto& a = out_attrs_[i];
      if (a.n_dims >= 3) {
        if (a.dims[1] == 4 || a.dims[2] == 4) idx_boxes_ = i;
        if (a.dims[1] == 1 || a.dims[2] == 1) idx_scores_ = i;
      }
    }

    for (int i = 0; i < io_num_.n_output; i++) {
      rknn_output o{};
      o.index = i;
      o.want_float = 1;
      o.is_prealloc = 1;

      // allocate float buffer by numel
      size_t numel = 1;
      for (int k = 0; k < out_attrs_[i].n_dims; k++) numel *= (size_t)out_attrs_[i].dims[k];
      out_buf_f32_[i].resize(numel);

      o.buf  = out_buf_f32_[i].data();
      o.size = out_buf_f32_[i].size() * sizeof(float);

      outs_[i] = o;
    }
  }

private:
  rknn_context ctx_ = 0;
  rknn_input_output_num io_num_{};
  rknn_tensor_attr in_attr_{};
  vector<rknn_tensor_attr> out_attrs_;

  vector<rknn_output> outs_;
  vector<vector<float>> out_buf_f32_;

  int idx_boxes_ = -1;
  int idx_scores_ = -1;
};

static vector<string> list_files(const string& dir) {
  vector<string> files;
  DIR* d = opendir(dir.c_str());
  if (!d) return files;
  dirent* e;
  while ((e = readdir(d)) != NULL) {
    string n = e->d_name;
    if (n == "." || n == "..") continue;
    files.push_back(n);
  }
  closedir(d);
  sort(files.begin(), files.end());
  return files;
}

static string model_path_from(const AppConfig& cfg) {
  // models/fp16/yolov8s_fp16.rknn
  // models/int8/yolov8s_hybrid_int8_presetB.rknn
  if (cfg.backend == "fp16") {
    return string("models/fp16/yolov8") + cfg.size + "_fp16.rknn";
  } else {
    return string("models/int8/yolov8") + cfg.size + "_hybrid_int8_presetB.rknn";
  }
}

int main() {
  AppConfig cfg;

  mutex cfg_mtx;
  atomic<bool> reload_req{true};
  atomic<bool> stop{false};

  // 최신 프레임/최신 JPEG 공유 (Queue=1)
  mutex frame_mtx;
  Mat latest_frame;
  atomic<uint64_t> frame_seq{0};

  mutex jpeg_mtx;
  vector<uchar> latest_jpeg;

  condition_variable jpeg_cv;
  atomic<uint64_t> jpeg_seq{0};

  // 상태
  atomic<double> last_infer_ms{0.0};
  atomic<int> last_det_count{0};

  // ---- Video decode thread ----
  thread th_decode([&](){
    VideoCapture cap;
    string opened;

    double src_fps = 30.0;
    double frame_ms = 1000.0 / src_fps;
    double next_t = now_ms();

    while (!stop.load()) {
      string v;
      {
        lock_guard<mutex> lk(cfg_mtx);
        v = cfg.video;
      }
      string path = string("videos/") + v;

      if (opened != path) {
        cap.release();
        cap.open(path);
        opened = path;
        if (!cap.isOpened()) {
          cerr << "[decode] failed to open video: " << path << "\n";
          this_thread::sleep_for(chrono::milliseconds(300));
          continue;
        }
        cerr << "[decode] opened: " << path << "\n";

        // FPS 기반 타이밍 초기화
        src_fps = cap.get(cv::CAP_PROP_FPS);
        if (!(src_fps > 1.0 && src_fps < 240.0)) src_fps = 30.0;
        frame_ms = 1000.0 / src_fps;
        next_t = now_ms();
        cerr << "[decode] src_fps=" << src_fps << " frame_ms=" << frame_ms << "\n";
      }

      Mat frame;
      if (!cap.read(frame)) {
        cap.set(CAP_PROP_POS_FRAMES, 0);
        next_t = now_ms(); // 루프 재시작시 타이밍 리셋
        continue;
      }

      {
        lock_guard<mutex> lk(frame_mtx);
        latest_frame = frame; // overwrite (Queue=1)
        frame_seq.fetch_add(1, std::memory_order_release);
      }

      // 원본 FPS에 맞춰 sleep (배속 방지)
      next_t += frame_ms;
      double wait = next_t - now_ms();
      if (wait < -2000.0) { next_t = now_ms(); wait = 0; }
      sleep_ms(wait);
    }
  });
  // ---- Inference thread ----
  thread th_infer([&](){
    ModelRunner runner;
    string loaded_model;
    uint64_t last_seq = 0;

    uint64_t infer_seq = 0;

    while (!stop.load()) {
      if (reload_req.load()) {
        AppConfig c;
        {
          lock_guard<mutex> lk(cfg_mtx);
          c = cfg;
        }
        string mp = model_path_from(c);
        if (mp != loaded_model) {
          cerr << "[infer] loading model: " << mp << "\n";
          if (!runner.load(mp)) {
            cerr << "[infer] model load failed.\n";
            this_thread::sleep_for(chrono::milliseconds(500));
            continue;
          }
          loaded_model = mp;
        }
        reload_req.store(false);
      }

      uint64_t cur_seq = frame_seq.load(std::memory_order_acquire);
      if (cur_seq == last_seq) {
        this_thread::sleep_for(chrono::milliseconds(2));
        continue;
      }
      last_seq = cur_seq;

      Mat frame;
      {
        lock_guard<mutex> lk(frame_mtx);
        if (latest_frame.empty()) {
          this_thread::sleep_for(chrono::milliseconds(5));
          continue;
        }
        frame = latest_frame.clone();
      }

      AppConfig c;
      {
        lock_guard<mutex> lk(cfg_mtx);
        c = cfg;
      }

      double infer_ms = 0.0;
      auto dets = runner.infer(frame, c.rgb, c.conf, c.nms, infer_ms);
      last_infer_ms.store(infer_ms);
      last_det_count.store((int)dets.size());

      // overlay on original 1920x1080
      for (auto& d : dets) {
        rectangle(frame, d.box, Scalar(0,255,0), 2);
        char buf[64];
        snprintf(buf, sizeof(buf), "Person %.2f", d.confidence);
        int x = max(0, (int)d.box.x);
        int y = max(0, (int)d.box.y - 5);
        putText(frame, buf, Point(x, y), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
      }

      infer_seq++;
      bool do_encode = (infer_seq % c.stream_skip) == 0;
      if (!do_encode) continue;
      int skip = c.stream_skip;
      if (skip < 1) skip = 1;
      if ((infer_seq % (uint64_t)skip) != 0) {
        continue;
      }

      // encode jpeg
      vector<int> params = {IMWRITE_JPEG_QUALITY, c.jpeg_quality};
      vector<uchar> jpeg;

      Mat view;
      if (frame.cols > 960) {
        resize(frame, view, Size(960, frame.rows * 960 / frame.cols));
      } else {
        view = frame;
      }

      bool ok = imencode(".jpg", view, jpeg, params);

      if (ok && !jpeg.empty()) {
        lock_guard<mutex> lk(jpeg_mtx);
        latest_jpeg.swap(jpeg);
      }
      jpeg_seq.fetch_add(1, std::memory_order_release);
      jpeg_cv.notify_all();
    }

    runner.unload();
  });

  // ---- HTTP server ----
  httplib::Server svr;

  // 정적 파일(간단 UI)
  auto serve_file = [&](const string& path, const string& content_type, httplib::Response& res){
    ifstream f(path, ios::binary);
    if (!f) { res.status = 404; return; }
    string body((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
    res.set_content(body, content_type.c_str());
  };

  svr.Get("/", [&](const httplib::Request&, httplib::Response& res){
    serve_file("web/index.html", "text/html; charset=utf-8", res);
  });
  svr.Get("/app.js", [&](const httplib::Request&, httplib::Response& res){
    serve_file("web/app.js", "application/javascript; charset=utf-8", res);
  });
  svr.Get("/style.css", [&](const httplib::Request&, httplib::Response& res){
    serve_file("web/style.css", "text/css; charset=utf-8", res);
  });

  // 상태
  svr.Get("/api/status", [&](const httplib::Request&, httplib::Response& res){
    AppConfig c;
    {
      lock_guard<mutex> lk(cfg_mtx);
      c = cfg;
    }
    char buf[512];
    snprintf(buf, sizeof(buf),
      "{"
      "\"backend\":\"%s\",\"size\":\"%s\",\"rgb\":%s,"
      "\"conf\":%.3f,\"nms\":%.3f,\"video\":\"%s\","
      "\"jpeg_quality\":%d,"
      "\"infer_ms\":%.2f,\"det_count\":%d"
      "}",
      c.backend.c_str(), c.size.c_str(), c.rgb ? "true":"false",
      c.conf, c.nms, c.video.c_str(),
      c.jpeg_quality,
      last_infer_ms.load(), last_det_count.load()
    );
    res.set_content(buf, "application/json; charset=utf-8");
  });

  // videos 목록
  svr.Get("/api/videos", [&](const httplib::Request&, httplib::Response& res){
    auto files = list_files("videos");
    string out = "[";
    for (size_t i=0;i<files.size();i++){
      out += "\"" + files[i] + "\"";
      if (i+1<files.size()) out += ",";
    }
    out += "]";
    res.set_content(out, "application/json; charset=utf-8");
  });

  // 설정 변경 (쿼리스트링)
  // /api/set?backend=fp16&size=s&rgb=1&conf=0.2&nms=0.45&video=test.mp4
  svr.Get("/api/set", [&](const httplib::Request& req, httplib::Response& res){
    bool need_reload = false;
    {
      lock_guard<mutex> lk(cfg_mtx);

      if (req.has_param("backend")) {
        string v = req.get_param_value("backend");
        if (v == "fp16" || v == "int8") {
          if (cfg.backend != v) need_reload = true;
          cfg.backend = v;
        }
      }
      if (req.has_param("size")) {
        string v = req.get_param_value("size");
        if (v=="n"||v=="s"||v=="m"||v=="l") {
          if (cfg.size != v) need_reload = true;
          cfg.size = v;
        }
      }
      if (req.has_param("rgb")) {
        string v = req.get_param_value("rgb");
        bool b = (v=="1"||v=="true"||v=="on");
        cfg.rgb = b;
      }
      if (req.has_param("conf")) cfg.conf = (float)atof(req.get_param_value("conf").c_str());
      if (req.has_param("nms"))  cfg.nms  = (float)atof(req.get_param_value("nms").c_str());
      if (req.has_param("video")) cfg.video = req.get_param_value("video");

      if (req.has_param("jpgq")) {
        int q = atoi(req.get_param_value("jpgq").c_str());
        if (q < 30) q = 30;
        if (q > 95) q = 95;
        cfg.jpeg_quality = q;
      }

      if (req.has_param("skip")) {
        int v = atoi(req.get_param_value("skip").c_str());
        if (v < 1) v = 1;
        if (v > 10) v = 10;
        cfg.stream_skip = v;
      }
    }

    if (need_reload) reload_req.store(true);

    res.set_content("{\"ok\":true}", "application/json; charset=utf-8");
  });

  // MJPEG 스트림
svr.Get("/stream.mjpg", [&](const httplib::Request&, httplib::Response& res){
  res.set_header("Cache-Control", "no-cache");
  res.set_header("Pragma", "no-cache");
  // res.set_header("Connection", "close"); // 필요하면 켜도 됨

  res.set_content_provider(
    "multipart/x-mixed-replace; boundary=frame",
    [&](size_t, httplib::DataSink& sink) {

      uint64_t last_sent = 0;

      while (!stop.load()) {
        vector<uchar> jpeg;

        // 새 JPEG가 나올 때까지 대기
        {
          unique_lock<mutex> lk(jpeg_mtx);
          jpeg_cv.wait_for(lk, chrono::milliseconds(500), [&]{
            return stop.load() || jpeg_seq.load(std::memory_order_acquire) != last_sent;
          });

          if (stop.load()) break;

          uint64_t cur = jpeg_seq.load(std::memory_order_acquire);
          if (cur == last_sent) continue;  // timeout but no new frame
          last_sent = cur;

          jpeg = latest_jpeg; // copy
        }

        if (!jpeg.empty()) {
          string header =
            "--frame\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: " + to_string(jpeg.size()) + "\r\n\r\n";

          if (!sink.write(header.data(), header.size())) break;
          if (!sink.write((const char*)jpeg.data(), jpeg.size())) break;
          if (!sink.write("\r\n", 2)) break;
        }
      }

      sink.done();
      return true;
    }
  );
});

  cout << "Server listening on 0.0.0.0:8080\n";
  svr.listen("0.0.0.0", 8080);

  // stop
  stop.store(true);
  if (th_decode.joinable()) th_decode.join();
  if (th_infer.joinable()) th_infer.join();
  return 0;
}
