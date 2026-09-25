import React, { useState, useEffect } from 'react';
import { Play, Square, Pause, PlayCircle, Settings2, Video, Camera, AlertCircle } from 'lucide-react';
import { MetricCard } from '../common/MetricCard';
import { TrackTable } from './TrackTable';
import { ClassDistribution } from './ClassDistribution';
import { api } from '../../api/client';

export function LiveDetection({
  streamState,
  config,
  onStart,
  onStop,
  onPause,
  onResume,
}) {
  const { status, currentFrame, error, isPaused, isStreaming } = streamState;

  // Stream controls
  const [sourceType, setSourceType] = useState('webcam'); // 'webcam' or 'video'
  const [cameraIndex, setCameraIndex] = useState(0);
  const [videoPath, setVideoPath] = useState('data/samples/sample_real_bus.mp4');
  const [confidence, setConfidence] = useState(config?.confidence_threshold || 0.35);
  const [iou, setIou] = useState(config?.iou_threshold || 0.45);
  const [fpsLimit, setFpsLimit] = useState(15);
  const [cameraProbe, setCameraProbe] = useState(null);
  const [probing, setProbing] = useState(false);

  // Probe camera when switching to webcam
  useEffect(() => {
    if (sourceType === 'webcam') {
      setProbing(true);
      api.probeWebcam(cameraIndex)
        .then((res) => setCameraProbe(res))
        .catch(() => setCameraProbe({ available: false, error: 'Probe failed' }))
        .finally(() => setProbing(false));
    }
  }, [sourceType, cameraIndex]);

  const handleStart = () => {
    onStart({
      source: sourceType,
      cameraIndex: Number(cameraIndex),
      path: sourceType === 'video' ? videoPath : null,
      fpsLimit: Number(fpsLimit),
      conf: Number(confidence),
      iou: Number(iou),
      annotate: true,
    });
  };

  // Derive class counts from frame
  const classCounts = {};
  if (currentFrame?.tracks && currentFrame.tracks.length > 0) {
    currentFrame.tracks.forEach((t) => {
      classCounts[t.class_name] = (classCounts[t.class_name] || 0) + 1;
    });
  } else if (currentFrame?.detections && currentFrame.detections.length > 0) {
    currentFrame.detections.forEach((d) => {
      classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1;
    });
  }

  // Pre-configured video sample paths
  const sampleVideos = [
    { label: 'Real Bus Street (data/samples/sample_real_bus.mp4)', path: 'data/samples/sample_real_bus.mp4' },
    { label: 'Synthetic Shapes (data/samples/sample_synthetic_shapes.mp4)', path: 'data/samples/sample_synthetic_shapes.mp4' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
        <MetricCard
          title="Streaming FPS"
          value={currentFrame?.fps ? currentFrame.fps.toFixed(1) : (isStreaming ? '...' : 0)}
          unit="fps"
          subtitle={isStreaming ? (isPaused ? 'Stream Paused' : 'Live Broadcast') : 'Offline'}
          color="#38bdf8"
        />
        <MetricCard
          title="Active Tracks"
          value={currentFrame?.active_tracks !== undefined ? currentFrame.active_tracks : 0}
          subtitle="Entities in current frame"
          color="#4ade80"
        />
        <MetricCard
          title="Cumulative Unique"
          value={currentFrame?.unique_tracks !== undefined ? currentFrame.unique_tracks : 0}
          subtitle="Persistent track IDs"
          color="#a855f7"
        />
        <MetricCard
          title="Inference Latency"
          value={currentFrame?.inference_time_ms !== undefined ? currentFrame.inference_time_ms.toFixed(1) : '--'}
          unit="ms"
          subtitle={`Tracking: ${currentFrame?.tracking_time_ms ? currentFrame.tracking_time_ms.toFixed(1) + 'ms' : '--'}`}
          color="#f59e0b"
        />
      </div>

      {/* Main Workspace Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 340px', gap: '20px' }}>
        {/* Left: Video / Stream Display */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div
            style={{
              backgroundColor: '#0f172a',
              borderRadius: '12px',
              border: '1px solid #334155',
              overflow: 'hidden',
              minHeight: '480px',
              display: 'flex',
              flexDirection: 'column',
              position: 'relative',
            }}
          >
            {/* Stream Header Bar */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '12px 16px',
                borderBottom: '1px solid #1e293b',
                backgroundColor: '#1e293b',
                fontSize: '13px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: isStreaming ? '#10b981' : '#64748b' }} />
                <span style={{ fontWeight: 600, color: '#f8fafc' }}>
                  {sourceType === 'webcam' ? `Camera Index ${cameraIndex}` : `Video: ${videoPath.split('/').pop()}`}
                </span>
              </div>
              <div style={{ color: '#94a3b8', fontSize: '12px', fontFamily: 'monospace' }}>
                Frame: #{currentFrame?.frame_number || 0}
              </div>
            </div>

            {/* Video Viewport / Frame */}
            <div
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '16px',
                backgroundColor: '#020617',
              }}
            >
              {error ? (
                <div style={{ textAlign: 'center', color: '#f87171', padding: '32px' }}>
                  <AlertCircle size={40} style={{ margin: '0 auto 12px' }} />
                  <div style={{ fontWeight: 600, fontSize: '15px' }}>Stream Interrupted</div>
                  <div style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>{error}</div>
                </div>
              ) : currentFrame?.annotated_frame ? (
                <img
                  src={`data:image/jpeg;base64,${currentFrame.annotated_frame}`}
                  alt="Live Vision Stream"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '520px',
                    objectFit: 'contain',
                    borderRadius: '6px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.4)',
                  }}
                />
              ) : isStreaming ? (
                <div style={{ textAlign: 'center', color: '#94a3b8' }}>
                  <div style={{ fontSize: '14px', marginBottom: '8px' }}>Initializing media stream pipeline...</div>
                  <div style={{ fontSize: '12px', color: '#64748b' }}>Awaiting first processed frame from backend</div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', color: '#64748b', padding: '48px 24px' }}>
                  <Video size={48} style={{ margin: '0 auto 16px', opacity: 0.4 }} />
                  <div style={{ fontSize: '16px', fontWeight: 600, color: '#94a3b8' }}>
                    Stream Standby
                  </div>
                  <div style={{ fontSize: '13px', marginTop: '6px', maxWidth: '380px' }}>
                    Select an input mode on the right and click <strong>Start Stream</strong> to begin real-time inference and ByteTrack tracking.
                  </div>
                </div>
              )}
            </div>

            {/* Stream Control Bar */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '12px 16px',
                borderTop: '1px solid #1e293b',
                backgroundColor: '#1e293b',
              }}
            >
              <div style={{ display: 'flex', gap: '8px' }}>
                {!isStreaming ? (
                  <button
                    onClick={handleStart}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '8px 16px',
                      borderRadius: '6px',
                      border: 'none',
                      backgroundColor: '#0284c7',
                      color: '#fff',
                      fontWeight: 600,
                      fontSize: '13px',
                      cursor: 'pointer',
                    }}
                  >
                    <Play size={15} /> Start Stream
                  </button>
                ) : (
                  <>
                    <button
                      onClick={onStop}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        padding: '8px 16px',
                        borderRadius: '6px',
                        border: 'none',
                        backgroundColor: '#ef4444',
                        color: '#fff',
                        fontWeight: 600,
                        fontSize: '13px',
                        cursor: 'pointer',
                      }}
                    >
                      <Square size={15} /> Stop Stream
                    </button>

                    {isPaused ? (
                      <button
                        onClick={onResume}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          padding: '8px 14px',
                          borderRadius: '6px',
                          border: '1px solid #334155',
                          backgroundColor: '#334155',
                          color: '#fff',
                          fontWeight: 500,
                          fontSize: '13px',
                          cursor: 'pointer',
                        }}
                      >
                        <PlayCircle size={15} /> Resume
                      </button>
                    ) : (
                      <button
                        onClick={onPause}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          padding: '8px 14px',
                          borderRadius: '6px',
                          border: '1px solid #334155',
                          backgroundColor: '#334155',
                          color: '#fff',
                          fontWeight: 500,
                          fontSize: '13px',
                          cursor: 'pointer',
                        }}
                      >
                        <Pause size={15} /> Pause
                      </button>
                    )}
                  </>
                )}
              </div>

              <div style={{ fontSize: '12px', color: '#94a3b8' }}>
                Total Detections: <span style={{ fontWeight: 600, color: '#f8fafc' }}>{currentFrame?.detections?.length || 0}</span>
              </div>
            </div>
          </div>

          {/* Active Tracks Table */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '16px' }}>
            <h3 style={{ margin: '0 0 12px', fontSize: '14px', fontWeight: 600, color: '#f8fafc' }}>
              Active Entity Telemetry (ByteTrack)
            </h3>
            <TrackTable tracks={currentFrame?.tracks || []} />
          </div>
        </div>

        {/* Right: Stream Parameter & Source Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Source Selection Panel */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
              Input Source Configuration
            </h3>

            {/* Source Tab Toggle */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '16px' }}>
              <button
                onClick={() => setSourceType('webcam')}
                disabled={isStreaming}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  border: 'none',
                  backgroundColor: sourceType === 'webcam' ? '#0284c7' : '#334155',
                  color: '#fff',
                  fontWeight: 600,
                  fontSize: '12px',
                  cursor: isStreaming ? 'not-allowed' : 'pointer',
                  opacity: isStreaming && sourceType !== 'webcam' ? 0.5 : 1,
                }}
              >
                <Camera size={14} /> Camera
              </button>
              <button
                onClick={() => setSourceType('video')}
                disabled={isStreaming}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  border: 'none',
                  backgroundColor: sourceType === 'video' ? '#0284c7' : '#334155',
                  color: '#fff',
                  fontWeight: 600,
                  fontSize: '12px',
                  cursor: isStreaming ? 'not-allowed' : 'pointer',
                  opacity: isStreaming && sourceType !== 'video' ? 0.5 : 1,
                }}
              >
                <Video size={14} /> Video File
              </button>
            </div>

            {sourceType === 'webcam' ? (
              <div>
                <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '6px' }}>
                  Camera Hardware Index:
                </label>
                <input
                  type="number"
                  min="0"
                  max="10"
                  value={cameraIndex}
                  disabled={isStreaming}
                  onChange={(e) => setCameraIndex(parseInt(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    color: '#f8fafc',
                    fontSize: '13px',
                    marginBottom: '10px',
                    boxSizing: 'border-box',
                  }}
                />
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>
                  {probing ? (
                    'Probing camera index...'
                  ) : cameraProbe?.available ? (
                    <span style={{ color: '#34d399' }}>✓ Camera index {cameraIndex} is available</span>
                  ) : (
                    <span style={{ color: '#fbbf24' }}>Notice: Camera {cameraIndex} reported offline or busy</span>
                  )}
                </div>
              </div>
            ) : (
              <div>
                <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '6px' }}>
                  Select Sample Video or Custom Relative Path:
                </label>
                <select
                  value={videoPath}
                  disabled={isStreaming}
                  onChange={(e) => setVideoPath(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 10px',
                    borderRadius: '6px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    color: '#f8fafc',
                    fontSize: '12px',
                    marginBottom: '10px',
                    boxSizing: 'border-box',
                  }}
                >
                  {sampleVideos.map((sv) => (
                    <option key={sv.path} value={sv.path}>{sv.label}</option>
                  ))}
                </select>

                <input
                  type="text"
                  value={videoPath}
                  disabled={isStreaming}
                  onChange={(e) => setVideoPath(e.target.value)}
                  placeholder="data/samples/sample_real_bus.mp4"
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    color: '#f8fafc',
                    fontSize: '12px',
                    fontFamily: 'monospace',
                    boxSizing: 'border-box',
                  }}
                />
              </div>
            )}
          </div>

          {/* Vision Inference Parameters */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
              Inference Thresholds
            </h3>

            {/* Confidence Slider */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                <span style={{ color: '#94a3b8' }}>Confidence Filter</span>
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
            <div style={{ marginBottom: '16px' }}>
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

            {/* FPS Limit */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                <span style={{ color: '#94a3b8' }}>Target FPS Rate</span>
                <span style={{ fontWeight: 600, color: '#38bdf8', fontFamily: 'monospace' }}>
                  {fpsLimit} fps
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="30"
                step="1"
                value={fpsLimit}
                disabled={isStreaming}
                onChange={(e) => setFpsLimit(parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          </div>

          {/* Class Distribution Card */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
              Active Class Breakdown
            </h3>
            <ClassDistribution classCounts={classCounts} />
          </div>
        </div>
      </div>
    </div>
  );
}
