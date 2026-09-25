import React, { useState } from 'react';
import { Upload, Image as ImageIcon, Play, AlertCircle, CheckCircle2 } from 'lucide-react';
import { BoundingBoxCanvas } from './BoundingBoxCanvas';
import { MetricCard } from '../common/MetricCard';
import { ClassDistribution } from './ClassDistribution';
import { api } from '../../api/client';

export function ImageDetection({ config }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [confidence, setConfidence] = useState(config?.confidence_threshold || 0.35);
  const [iou, setIou] = useState(config?.iou_threshold || 0.45);
  const [loading, setLoading] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [error, setError] = useState(null);

  // Quick samples available on disk
  const samplePresets = [
    { name: 'City Bus Scenario (bus.jpg)', url: '/data/samples/bus.jpg', filename: 'bus.jpg' },
    { name: 'People / Sports (zidane.jpg)', url: '/data/samples/zidane.jpg', filename: 'zidane.jpg' },
  ];

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setDetectionResult(null);
      setError(null);
    }
  };

  const handleLoadSample = async (sample) => {
    try {
      setLoading(true);
      setError(null);
      // Fetch sample image from local static samples
      const res = await fetch(sample.url);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: blob.type || 'image/jpeg' });
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setDetectionResult(null);
    } catch (err) {
      setError(`Failed to load sample image: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRunDetection = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const result = await api.detectImage(selectedFile, confidence, iou);
      setDetectionResult(result);
    } catch (err) {
      setError(err.message || 'Detection failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
        <MetricCard
          title="Total Detections"
          value={detectionResult ? detectionResult.total_detections : 0}
          subtitle="Objects identified"
          color="#38bdf8"
        />
        <MetricCard
          title="Inference Time"
          value={detectionResult?.inference_time_ms !== undefined ? detectionResult.inference_time_ms.toFixed(1) : '--'}
          unit="ms"
          subtitle="Forward pass latency"
          color="#f59e0b"
        />
        <MetricCard
          title="Image Resolution"
          value={detectionResult ? `${detectionResult.image_width}×${detectionResult.image_height}` : '--'}
          subtitle="Source dimensions"
          color="#a855f7"
        />
        <MetricCard
          title="Inference Device"
          value={detectionResult ? detectionResult.device : (config?.device || 'Auto')}
          subtitle="Execution processor"
          color="#4ade80"
        />
      </div>

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 340px', gap: '20px' }}>
        {/* Left: Canvas Viewport */}
        <div
          style={{
            backgroundColor: '#0f172a',
            borderRadius: '12px',
            border: '1px solid #334155',
            overflow: 'hidden',
            minHeight: '520px',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Viewport Header */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '12px 16px',
              backgroundColor: '#1e293b',
              borderBottom: '1px solid #334155',
              fontSize: '13px',
            }}
          >
            <span style={{ fontWeight: 600, color: '#f8fafc' }}>
              {selectedFile ? `File: ${selectedFile.name}` : 'No Image Loaded'}
            </span>
            {detectionResult && (
              <span style={{ color: '#34d399', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px' }}>
                <CheckCircle2 size={14} /> Inference Complete
              </span>
            )}
          </div>

          {/* Canvas Viewport */}
          <div
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '16px',
              backgroundColor: '#020617',
              minHeight: '440px',
            }}
          >
            {error ? (
              <div style={{ textAlign: 'center', color: '#f87171', padding: '32px' }}>
                <AlertCircle size={40} style={{ margin: '0 auto 12px' }} />
                <div style={{ fontWeight: 600, fontSize: '15px' }}>Inference Error</div>
                <div style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>{error}</div>
              </div>
            ) : previewUrl ? (
              <BoundingBoxCanvas
                imageSrc={previewUrl}
                detections={detectionResult?.detections || []}
                width={detectionResult?.image_width || 640}
                height={detectionResult?.image_height || 480}
                showLabels={true}
                showConfidence={true}
                showTracks={false}
              />
            ) : (
              <div style={{ textAlign: 'center', color: '#64748b', padding: '48px 24px' }}>
                <ImageIcon size={48} style={{ margin: '0 auto 16px', opacity: 0.4 }} />
                <div style={{ fontSize: '16px', fontWeight: 600, color: '#94a3b8' }}>
                  Image Workspace Ready
                </div>
                <div style={{ fontSize: '13px', marginTop: '6px', maxWidth: '380px' }}>
                  Upload an image (JPG, PNG, WEBP, BMP) or pick a sample preset on the right, then run detection.
                </div>
              </div>
            )}
          </div>

          {/* Detections Breakdown footer */}
          {detectionResult?.detections && detectionResult.detections.length > 0 && (
            <div style={{ padding: '12px 16px', backgroundColor: '#1e293b', borderTop: '1px solid #334155' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#cbd5e1', marginBottom: '8px' }}>
                Detected Entities ({detectionResult.detections.length}):
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', maxHeight: '120px', overflowY: 'auto' }}>
                {detectionResult.detections.map((det, idx) => (
                  <span
                    key={idx}
                    style={{
                      padding: '4px 8px',
                      borderRadius: '4px',
                      backgroundColor: '#0f172a',
                      border: '1px solid #334155',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      color: '#38bdf8',
                    }}
                  >
                    {det.class_name} ({Math.round(det.confidence * 100)}%)
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Controls Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* File Upload Box */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
              Load Image
            </h3>

            <label
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '24px 16px',
                borderRadius: '8px',
                border: '2px dashed #334155',
                cursor: 'pointer',
                backgroundColor: '#0f172a',
                transition: 'all 0.2s',
                marginBottom: '16px',
              }}
            >
              <Upload size={24} color="#38bdf8" style={{ marginBottom: '8px' }} />
              <span style={{ fontSize: '13px', fontWeight: 500, color: '#f8fafc' }}>Choose an image</span>
              <span style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>JPG, PNG, WEBP, BMP up to 15MB</span>
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp,image/bmp"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </label>

            {/* Presets */}
            <div style={{ fontSize: '12px', fontWeight: 500, color: '#94a3b8', marginBottom: '8px' }}>
              Or quick select test sample:
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {samplePresets.map((sp) => (
                <button
                  key={sp.filename}
                  onClick={() => handleLoadSample(sp)}
                  disabled={loading}
                  style={{
                    padding: '8px 12px',
                    borderRadius: '6px',
                    border: '1px solid #334155',
                    backgroundColor: '#0f172a',
                    color: '#e2e8f0',
                    fontSize: '12px',
                    textAlign: 'left',
                    cursor: 'pointer',
                  }}
                >
                  {sp.name}
                </button>
              ))}
            </div>
          </div>

          {/* Parameters & Run Action */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
              Inference Parameters
            </h3>

            {/* Confidence Slider */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                <span style={{ color: '#94a3b8' }}>Confidence Threshold</span>
                <span style={{ fontWeight: 600, color: '#38bdf8', fontFamily: 'monospace' }}>
                  {confidence.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            {/* IoU Slider */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                <span style={{ color: '#94a3b8' }}>NMS IoU Threshold</span>
                <span style={{ fontWeight: 600, color: '#38bdf8', fontFamily: 'monospace' }}>
                  {iou.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={iou}
                onChange={(e) => setIou(parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunDetection}
              disabled={!selectedFile || loading}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                padding: '10px 16px',
                borderRadius: '6px',
                border: 'none',
                backgroundColor: !selectedFile || loading ? '#334155' : '#0284c7',
                color: '#fff',
                fontWeight: 600,
                fontSize: '13px',
                cursor: !selectedFile || loading ? 'not-allowed' : 'pointer',
              }}
            >
              <Play size={16} />
              {loading ? 'Running Inference...' : 'Run YOLO Detection'}
            </button>
          </div>

          {/* Class Breakdown */}
          {detectionResult?.class_counts && (
            <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
              <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
                Class Distribution
              </h3>
              <ClassDistribution classCounts={detectionResult.class_counts} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
