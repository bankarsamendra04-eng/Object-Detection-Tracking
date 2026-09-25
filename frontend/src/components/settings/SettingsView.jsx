import React, { useState } from 'react';
import { Sliders, Camera, Cpu, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { api } from '../../api/client';

export function SettingsView({ config }) {
  const [probeIndex, setProbeIndex] = useState(0);
  const [probeResult, setProbeResult] = useState(null);
  const [probing, setProbing] = useState(false);

  const handleProbeCamera = async () => {
    try {
      setProbing(true);
      const res = await api.probeWebcam(probeIndex);
      setProbeResult(res);
    } catch (err) {
      setProbeResult({ available: false, error: err.message });
    } finally {
      setProbing(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div>
        <h2 style={{ fontSize: '18px', fontWeight: 700, margin: 0, color: '#f8fafc' }}>
          Runtime Configuration & Hardware Diagnostics
        </h2>
        <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#94a3b8' }}>
          Verify current backend parameters, supported formats, and hardware devices
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px' }}>
        {/* Detection & Tracking Configuration */}
        <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
          <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sliders size={16} color="#38bdf8" /> Vision Engine Settings
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '13px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid #334155' }}>
              <span style={{ color: '#94a3b8' }}>Active Model Checkpoint:</span>
              <span style={{ color: '#38bdf8', fontFamily: 'monospace' }}>{config?.model_name || 'yolov8n.pt'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid #334155' }}>
              <span style={{ color: '#94a3b8' }}>Confidence Threshold:</span>
              <span style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{config?.confidence_threshold ?? 0.35}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid #334155' }}>
              <span style={{ color: '#94a3b8' }}>NMS IoU Threshold:</span>
              <span style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{config?.iou_threshold ?? 0.45}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid #334155' }}>
              <span style={{ color: '#94a3b8' }}>Multi-Object Tracker:</span>
              <span style={{ color: '#a855f7', fontFamily: 'monospace' }}>{config?.tracker_type || 'bytetrack.yaml'} (ByteTrack)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid #334155' }}>
              <span style={{ color: '#94a3b8' }}>Track High Thresh:</span>
              <span style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{config?.track_high_thresh ?? 0.5}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid #334155' }}>
              <span style={{ color: '#94a3b8' }}>Track Low Thresh:</span>
              <span style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{config?.track_low_thresh ?? 0.1}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid #334155' }}>
              <span style={{ color: '#94a3b8' }}>Track Match IoU Thresh:</span>
              <span style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{config?.track_match_thresh ?? 0.8}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid #334155' }}>
              <span style={{ color: '#94a3b8' }}>Track Persistence Buffer:</span>
              <span style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{config?.track_persistence_buffer ?? 30} frames</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid #334155' }}>
              <span style={{ color: '#94a3b8' }}>Execution Device:</span>
              <span style={{ color: '#4ade80', fontFamily: 'monospace' }}>{config?.device || 'Auto'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Environment:</span>
              <span style={{ color: '#e2e8f0' }}>{config?.environment || 'development'}</span>
            </div>
          </div>
        </div>

        {/* Camera Hardware Diagnostic Prober */}
        <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
          <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Camera size={16} color="#38bdf8" /> Camera Hardware Diagnostics
          </h3>

          <p style={{ margin: '0 0 16px', fontSize: '13px', color: '#94a3b8' }}>
            Test camera device availability through backend DirectShow/V4L2 probe without opening a stream.
          </p>

          <div style={{ display: 'flex', gap: '10px', marginBottom: '16px' }}>
            <input
              type="number"
              min="0"
              max="10"
              value={probeIndex}
              onChange={(e) => setProbeIndex(parseInt(e.target.value) || 0)}
              style={{
                width: '80px',
                padding: '8px 12px',
                borderRadius: '6px',
                backgroundColor: '#0f172a',
                border: '1px solid #334155',
                color: '#f8fafc',
                fontSize: '13px',
              }}
            />
            <button
              onClick={handleProbeCamera}
              disabled={probing}
              style={{
                flex: 1,
                padding: '8px 16px',
                borderRadius: '6px',
                border: 'none',
                backgroundColor: '#0284c7',
                color: '#fff',
                fontSize: '13px',
                fontWeight: 600,
                cursor: probing ? 'not-allowed' : 'pointer',
              }}
            >
              {probing ? 'Probing Device...' : `Probe Camera ${probeIndex}`}
            </button>
          </div>

          {probeResult && (
            <div
              style={{
                padding: '12px',
                borderRadius: '8px',
                backgroundColor: '#0f172a',
                border: `1px solid ${probeResult.available ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
                fontSize: '12px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, color: probeResult.available ? '#34d399' : '#fbbf24', marginBottom: '4px' }}>
                {probeResult.available ? <CheckCircle2 size={15} /> : <ShieldAlert size={15} />}
                {probeResult.available ? `Camera ${probeIndex} Available` : `Camera ${probeIndex} Unavailable`}
              </div>
              <div style={{ color: '#94a3b8' }}>
                Backend API verified device index without lock contention.
              </div>
            </div>
          )}
        </div>

        {/* Media Ingestion Support */}
        <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
          <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
            Supported File Codecs & Formats
          </h3>

          <div style={{ marginBottom: '14px' }}>
            <div style={{ fontSize: '12px', fontWeight: 600, color: '#cbd5e1', marginBottom: '6px' }}>
              Image Formats:
            </div>
            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              {(config?.supported_image_formats || ['.jpg', '.jpeg', '.png', '.webp', '.bmp']).map((fmt) => (
                <span
                  key={fmt}
                  style={{
                    padding: '3px 8px',
                    borderRadius: '4px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                    color: '#38bdf8',
                  }}
                >
                  {fmt}
                </span>
              ))}
            </div>
          </div>

          <div>
            <div style={{ fontSize: '12px', fontWeight: 600, color: '#cbd5e1', marginBottom: '6px' }}>
              Video Codecs / Containers:
            </div>
            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              {(config?.supported_video_formats || ['.mp4', '.avi', '.mov', '.mkv']).map((fmt) => (
                <span
                  key={fmt}
                  style={{
                    padding: '3px 8px',
                    borderRadius: '4px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                    color: '#a855f7',
                  }}
                >
                  {fmt}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
