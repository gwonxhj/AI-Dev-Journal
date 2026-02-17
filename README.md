# 권혁준 | Edge AI Inference Optimization Engineer

Edge 환경에서 AI 추론 성능을 분석하고 최적화하는 시스템 지향 개발자입니다.
모델 정확도 유지와 동시에 Latency 감소, 병목 분석, 멀티스레드 파이프라인 설계에 집중하고 있습니다.
 
⸻

## 🚀 Performance Highlights
	•	YOLOv8 FP32 → Hybrid INT8 적용 후 Latency 42ms → 21ms (약 50% 개선)
	•	RK3588 NPU 환경에서 INT8 양자화 시 출력 텐서 붕괴 문제 분석 및 Hybrid 설계
	•	Multi-thread Producer–Consumer 구조 적용 후 FPS 28 → 46 향상
	•	Preprocess / Inference / Postprocess 단계별 Latency Breakdown 분석
	•	NPU–CPU Fallback 조건 분석 및 병목 제거 경험

⸻

## 🎯 Core Focus
	•	Edge AI Inference Optimization
	•	Quantization Strategy (FP32 / FP16 / INT8 / Hybrid)
	•	Multi-threaded Real-time Pipeline
	•	NPU / GPU 기반 추론 성능 분석
	•	System-Level Bottleneck Debugging

⸻

## 🧠 Technical Experience

## 🔹 Inference Optimization
	•	YOLOv8 모델 ONNX 변환 및 TensorRT / RKNN 엔진 빌드
	•	FP16 vs INT8 vs Hybrid INT8 성능 비교 실험
	•	RKNN Toolkit 기반 레이어 단위 Hybrid Quantization 설계
	•	INT8 양자화 시 출력 텐서 붕괴 문제 디버깅 및 정확도 복구
	•	Latency Breakdown 도구 구현 (Stage별 측정)
	•	CPU affinity, Thread 분리, Queue 구조 개선 실험

⸻

## 🔹 Embedded / Edge Platforms
	•	Jetson Orin Nano / AGX Orin (TensorRT 기반 실시간 추론)
	•	Odroid M1 / M2 (RK3588S NPU 기반 Hybrid INT8 추론)
	•	Raspberry Pi 5 / Zero 2W
	•	Ubuntu 20.04 / 22.04 기반 시스템 튜닝

⸻

## 🔹 System Architecture
	•	Multi-thread Producer–Consumer 파이프라인 구현
	•	Video Decode → Preprocess → Inference → Postprocess 분리 구조
	•	Memory Reuse 전략 적용
	•	RTSP/MP4 기반 FFmpeg 영상 디코딩 파이프라인 구축

⸻

## 🔹 Programming
	•	C++ (멀티스레드 기반 추론 파이프라인)
	•	Python (모델 변환 및 실험 자동화)
	•	C (MCU 기반 제어)
	•	Shell Script (환경 자동화)

⸻

## 📂 Projects

⸻

## 1️⃣ Beach CCTV AI — Edge NPU Inference Optimization

**Odroid M2 (RK3588S) + YOLOv8 + RKNN Toolkit**

실시간 돌발행동 감지를 위한 Edge NPU 기반 추론 시스템 최적화 프로젝트.

**🔥 Optimization Work**
	•	YOLOv8 ONNX → RKNN 변환
	•	FP16 / INT8 / Hybrid INT8 성능 비교
	•	INT8 적용 시 출력 텐서 붕괴 현상 분석
	•	레이어 단위 Hybrid INT8 설계
	•	Multi-thread 추론 파이프라인 구성

**📊 Performance Result**

Precision	Latency (ms)	FPS
FP32	|| 42ms ||	23
FP16	29ms	34
Hybrid INT8	21ms	46

정확도 유지 + 약 50% Latency 개선

→ 프로젝트 보러가기￼

⸻

## 2️⃣ Fire Detection Self-Driving Drone

**Jetson Orin Nano + YOLOv8 + TensorRT**

화재 탐지 및 마커 기반 착륙 자율 비행 시스템.

**System Engineering Focus**
	•	TensorRT 기반 FP16 엔진 최적화
	•	실시간 영상 스트림 기반 추론 파이프라인 설계
	•	AI 추론 결과와 비행 제어 시스템 연동

→ 프로젝트 보러가기￼

⸻

## 3️⃣ Gesture Motion RC Car

센서 기반 실시간 제어 시스템 구현 (Arduino + MPU-6050)

→ 프로젝트 보러가기￼

⸻

## 📫 Contact

Email: ksjm0417@naver.com
