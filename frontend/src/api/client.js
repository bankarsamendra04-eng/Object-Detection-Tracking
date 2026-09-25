/**
 * Centralized API Client for Computer Vision Platform
 * Handles REST requests, error formatting, and environment configuration.
 */

const API_BASE = '/api/v1';

class ApiError extends Error {
  constructor(message, status, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const headers = options.headers || {};

  if (!(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  try {
    const res = await fetch(url, { ...options, headers });
    const isJson = res.headers.get('content-type')?.includes('application/json');
    const data = isJson ? await res.json() : await res.text();

    if (!res.ok) {
      const errorMessage = typeof data === 'object' && data.detail 
        ? data.detail 
        : `Request failed with status ${res.status}`;
      throw new ApiError(errorMessage, res.status, data);
    }

    return data;
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(err.message || 'Network error connecting to backend service.', 0);
  }
}

export const api = {
  // System Health & Config
  getHealth: () => request('/health'),
  getRootStatus: () => request('/status'),
  getConfig: () => request('/config'),
  getSources: () => request('/sources'),
  probeWebcam: (cameraIndex = 0) => request(`/sources/webcam/probe?camera_index=${cameraIndex}`),

  // Object Detection
  detectImage: (file, confidence = null, iou = null) => {
    const formData = new FormData();
    formData.append('file', file);
    if (confidence !== null && confidence !== undefined) {
      formData.append('confidence', confidence.toString());
    }
    if (iou !== null && iou !== undefined) {
      formData.append('iou', iou.toString());
    }
    return request('/detect/image', {
      method: 'POST',
      body: formData,
    });
  },

  // Video File Tracking
  trackVideo: (file, confidence = null, iou = null, stride = 1) => {
    const formData = new FormData();
    formData.append('file', file);
    if (confidence !== null) formData.append('confidence', confidence.toString());
    if (iou !== null) formData.append('iou', iou.toString());
    formData.append('stride', stride.toString());
    return request('/track/video', {
      method: 'POST',
      body: formData,
    });
  },

  // Model Management
  listModels: () => request('/models'),
  getActiveModel: () => request('/models/active'),
  getModel: (modelId) => request(`/models/${encodeURIComponent(modelId)}`),
  validateModel: (modelId) => request(`/models/${encodeURIComponent(modelId)}/validate`, { method: 'POST' }),
  switchModel: (modelId) => request(`/models/${encodeURIComponent(modelId)}/switch`, { method: 'POST' }),

  // Analytics
  getAnalyticsReport: (sessionId = 'default_session') => request(`/analytics/report?session_id=${encodeURIComponent(sessionId)}`),
  getAnalyticsSnapshot: (sessionId = 'default_session') => request(`/analytics/snapshot?session_id=${encodeURIComponent(sessionId)}`),
  resetAnalytics: (sessionId = 'default_session') => request(`/analytics/reset?session_id=${encodeURIComponent(sessionId)}`, { method: 'POST' }),
};

export default api;
