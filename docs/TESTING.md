# Testing Architecture & Quality Assurance Guide

This document describes the automated testing strategy, the 8-level test pyramid, test execution commands, and verified results for the **Real-Time Object Detection & Tracking Platform**.

---

## 1. Testing Strategy & Multi-Level Pyramid

The quality assurance architecture is structured into 8 distinct validation tiers:

```
                  ▲
                 / \     Level 8: Real Smoke & Concurrency (Multi-threaded & live hardware)
                /   \    Level 7: Failure Injection (Corrupt media, missing models, traversal)
               /     \   Level 6: End-to-End System Integration (Multi-component pipelines)
              /       \  Level 5: Frontend State Machine & Telemetry (Node unit testing)
             /         \ Level 4: WebSocket Protocol (State transitions, queue backpressure)
            /           \Level 3: REST API Contracts (Schema validation, HTTP status codes)
           /             \Level 2: Pipeline Integration (Detector -> ByteTrack -> Analytics)
          /_______________\Level 1: Unit Testing & Mathematical Bounds (Geometry, deques, models)
```

---

## 2. Test Suite Inventory

| Test Module | Level | Focus / Coverage | Test Count |
| :--- | :--- | :--- | :--- |
| `tests/unit/test_detector.py` | Level 1 | 1x1 image inference, grayscale frames, confidence/IoU filtering, FP16 half-precision toggle. | 9 |
| `tests/unit/test_tracker.py` | Level 1 | ByteTrack two-stage association, trajectory ring buffers, track persistence buffers. | 11 |
| `tests/unit/test_analytics.py` | Level 1 | 2D vector cross-product geometry (`ccw`), forward/backward crossing, entity deduplication. | 10 |
| `tests/unit/test_input_sources.py`| Level 1 | ImageInput, VideoInput, WebcamInput, EOF handling, corrupt file rejection. | 13 |
| `tests/unit/test_model_manager.py`| Level 1 | Model registry YAML parsing, weights validation, dynamic hot-switching, cache clearing. | 17 |
| `tests/unit/test_dataset_pipeline.py`| Level 1 | Coordinate normalization, VisDrone conversion, dataset leakage checks. | 11 |
| `tests/unit/test_config.py` | Level 1 | Path traversal sanitization, CORS assembly, environment variable resolution. | 3 |
| `tests/api/test_endpoints.py` | Level 3 | REST status codes, image detection endpoint, video tracking, analytics report endpoints. | 19 |
| `tests/api/test_models_api.py` | Level 3 | Model registry listing, model switching endpoints, error responses. | 8 |
| `tests/api/test_websocket.py` | Level 4 | Connection handshake, start/stop/pause/resume commands, frame throttling. | 12 |
| `tests/e2e/test_step12_full_e2e.py`| Level 6 | Full end-to-end pipeline: InputSource -> Detector -> ByteTrack -> Analytics -> DB. | 8 |
| `tests/security/test_step13_security_hardening.py` | Level 7 | Upload size caps (15MB/50MB), 8192px image dimension limits, security headers. | 23 |
| `tests/qa/test_step14_qa_suite.py`| Level 7 | Edge case regression, zero-byte uploads, rapid reconnects, concurrent sessions. | 28 |
| `tests/performance/test_step15_resource_leak.py` | Level 8 | Sustained video streaming (25+ frames), queue backpressure, VRAM leak verification. | 2 |
| `tests/deployment/test_step16_docker_config.py` | Level 8 | Dockerfile best practices, docker-compose syntax, Nginx routing, .dockerignore rules. | 8 |
| `frontend/tests/frontend.test.js` | Level 5 | React state machine, connection lifecycle, telemetry meters, bbox geometry. | 12 |
| **Total Test Inventory** | — | **Full System Automated Quality Assurance** | **194 Tests** |

---

## 3. Test Execution Commands

### 3.1 Running the Full Backend Test Suite
```powershell
# In PowerShell (activate virtualenv first)
.\.venv\Scripts\Activate.ps1
pytest -q
```
**Latest Verified Output:**
```
============================ 182 passed in 17.60s =============================
```

### 3.2 Running the Frontend Test Suite
```powershell
cd frontend
npm test -- --run
cd ..
```
**Latest Verified Output:**
```
✔ Frontend Unit Tests: WebSocket Protocol & State Machine (2.82ms)
✔ Frontend Unit Tests: Data Calculations & Formatting (1.19ms)
ℹ tests 12 | pass 12 | fail 0
```

### 3.3 Running the Live Application Smoke Test
```powershell
.\.venv\Scripts\python tests/security/smoke_test_step13.py
```
**Verified Output:**
```
[1/7] Testing Security Headers on HTTP Responses... PASS
[2/7] Checking Config and Diagnostics for Secrets Leaks... PASS
[3/7] Testing Real Image Detection on bus.jpg... PASS (4 objects detected in 13.5ms)
[4/7] Testing Real Video Tracking on sample_real_bus.mp4... PASS (10 frames, 4 objects)
[5/7] Testing Real WebSocket Streaming with ByteTrack + Analytics... PASS
[6/7] Verifying WebSocket Disconnect Cleanup... PASS
[7/7] Testing Malicious Request Rejection... PASS
============================================================
ALL REAL APPLICATION SMOKE TESTS PASSED SUCCESSFULLY!
============================================================
```

### 3.4 Running the Resource Leak & Performance Test
```powershell
.\.venv\Scripts\python -m pytest tests/performance/test_step15_resource_leak.py -v
```
**Verified Output:**
```
tests/performance/test_step15_resource_leak.py::TestResourceLeak::test_sustained_streaming_resource_leak PASSED
tests/performance/test_step15_resource_leak.py::TestResourceLeak::test_repeated_static_inference_memory_stability PASSED
============================== 2 passed in 1.48s ==============================
```
