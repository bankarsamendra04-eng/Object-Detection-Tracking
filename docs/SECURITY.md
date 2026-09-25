# Security Architecture & Production Hardening Guide

This document details the security safeguards, input validation boundaries, and defensive architecture implemented during **Step 13** and reinforced across the platform.

---

## 1. Threat Model & Defensive Boundaries

The system is designed around defensive boundaries mitigating the primary Computer Vision and web application attack vectors:

```
[Untrusted Client Request]
       │
       ▼
[Security Headers Middleware]   ──> (Nosniff, Frame-Options, XSS, Referrer-Policy)
       │
       ▼
[Input Sanitizer & Boundary Check]
  ├── Path Traversal Defense     ──> (Strict PROJECT_ROOT anchoring, null-byte rejection)
  ├── Upload Dimension Limiter   ──> (Decompression bomb prevention: max 8192px)
  ├── File Size Verifier         ──> (Max 15 MB image, max 50 MB video)
  └── WebSocket Rate Limiter     ──> (Max 64 KB messages, max 10 concurrent sessions)
       │
       ▼
[Business Logic: YOLO / ByteTrack / SQLite]
```

---

## 2. Path Traversal & File System Defenses

Arbitrary file access via malicious file paths is prevented by `Settings.resolve_safe_path()` (`backend/app/core/config.py`):

```python
def resolve_safe_path(self, user_path: Union[str, Path], allowed_dir: Optional[Union[str, Path]] = None) -> Path:
    # 1. Reject null-byte injection
    if "\x00" in p_str:
        raise ValueError("Invalid null byte in path.")

    # 2. Reject UNC paths (e.g. \\attacker-smb\share)
    if p_str.startswith("\\\\") or p_str.startswith("//"):
        raise ValueError("UNC paths are not permitted.")

    # 3. Canonicalize and enforce strict boundary containment
    resolved = candidate.resolve()
    if not (resolved == base or base in resolved.parents):
        raise ValueError(f"Path traversal detected: {user_path}")
    return resolved
```

- **Rejection of Traversal Patterns:** Requests targeting `../../etc/passwd` or `..\..\Windows` are rejected with HTTP 400 / HTTP 404 before file I/O operations occur.
- **Allowed Directories:** File operations are restricted strictly to `PROJECT_ROOT`, `data/`, `data/uploads/`, and `outputs/`.

---

## 3. Media Ingestion & Upload Safeguards

To protect against Denial of Service (DoS) and decompression bombs:

1. **Maximum File Size Caps:**
   - Image uploads: Strictly limited to **15 MB** (`MAX_UPLOAD_IMAGE_SIZE_BYTES = 15728640`).
   - Video uploads: Strictly limited to **50 MB** (`MAX_UPLOAD_VIDEO_SIZE_BYTES = 52428800`).
   - Uploads exceeding these thresholds trigger immediate HTTP `413 Request Entity Too Large` without buffering to disk.
2. **Decompression Bomb Protection:**
   - Image dimensions are validated prior to NumPy allocation.
   - Images exceeding **8192 pixels** in width or height are rejected with HTTP 400.
3. **MIME-Type Allowlisting:**
   - Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.
   - Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.
   - Unrecognized file extensions trigger HTTP 400 with a structured error response.

---

## 4. WebSocket Streaming Defenses

1. **Payload Size Caps:** Client-to-server WebSocket messages are capped at **64 KB** (`MAX_WS_MESSAGE_SIZE_BYTES = 65536`). Oversized messages trigger immediate protocol disconnection.
2. **Concurrency Limiting:** Active simultaneous streaming sessions are limited to **10 concurrent clients** (`MAX_CONCURRENT_WS_SESSIONS = 10`), preventing server resource exhaustion.
3. **Graceful Teardown on Disconnect:** Unhandled network disconnections automatically trigger `StreamSession.close()`, releasing video capture handles and cancelling asynchronous tasks.

---

## 5. HTTP Security Headers

The FastAPI middleware injects production security headers on every HTTP response:
```http
X-Content-Type-Options: nosniff
X-Frame-Options: SAMEORIGIN
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
```

---

## 6. Container Security (Docker)

1. **Non-Root Execution:** The backend container defines and runs under an unprivileged user `appuser` (UID 1000). Root access inside the container is disallowed.
2. **Read-Only Model Mounts:** Default model checkpoints (`yolov8n.pt`) are mounted into containers as read-only volumes (`:ro`).
3. **Minimal Attack Surface:** Uses `python:3.11-slim-bookworm` and `nginx:1.27-alpine` base images, stripping development tools and package managers from runtime images.
