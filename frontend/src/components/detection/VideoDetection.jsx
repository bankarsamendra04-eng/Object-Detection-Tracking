import React, { useState } from 'react';
import { Film, Upload, Play, AlertCircle, CheckCircle2 } from 'lucide-react';
import { MetricCard } from '../common/MetricCard';
import { TrackTable } from './TrackTable';
import { ClassDistribution } from './ClassDistribution';
import { api } from '../../api/client';

export function VideoDetection({ config, onSwitchToLiveStream }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [confidence, setConfidence] = useState(config?.confidence_threshold || 0.35);
  const [iou, setIou] = useState(config?.iou_threshold || 0.45);
  const [stride, setStride] = useState(2);
  const [loading, setLoading] = useState(false);
  const [progressText, setProgressText] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
    }
  };

  const handleProcessVideo = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setProgressText('Uploading video to backend processing pipeline...');

    try {
      setProgressText('Executing ByteTrack multi-object tracking on video frames...');
      const res = await api.trackVideo(selectedFile, confidence, iou, stride);
      setResult(res);
      setProgressText('Video processing complete.');
    } catch (err) {
      setError(err.message || 'Video tracking failed.');
    } finally {
      setLoading(false);
    }
  };

  // Compile active tracks or cumulative tracks from result
  const tracks = result?.unique_objects || [];
  const classCounts = {};
  tracks.forEach((t) => {
    classCounts[t.class_name] = (classCounts[t.class_name] || 0) + 1;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
        <MetricCard
          title="Frames Processed"
          value={result?.frames_processed !== undefined ? result.frames_processed : 0}
          subtitle={`Total frames: ${result?.total_frames || '--'}`}
          color="#38bdf8"
        />
        <MetricCard
          title="Unique Entities"
          value={result?.total_unique_tracks !== undefined ? result.total_unique_tracks : 0}
          subtitle="Persistent tracks identified"
          color="#4ade80"
        />
        <MetricCard
          title="Processing Time"
          value={result?.processing_time_seconds !== undefined ? result.processing_time_seconds.toFixed(1) : '--'}
          unit="s"
          subtitle={`Effective FPS: ${result?.processing_fps ? result.processing_fps.toFixed(1) : '--'}`}
          color="#f59e0b"
        />
        <MetricCard
          title="Inference Device"
          value={config?.device || 'Auto'}
          subtitle="Hardware backend"
          color="#a855f7"
        />
      </div>

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 340px', gap: '20px' }}>
        {/* Left: Results Display & Telemetry */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div
            style={{
              backgroundColor: '#0f172a',
              borderRadius: '12px',
              border: '1px solid #334155',
              padding: '24px',
              minHeight: '400px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            {error ? (
              <div style={{ textAlign: 'center', color: '#f87171', padding: '32px' }}>
                <AlertCircle size={40} style={{ margin: '0 auto 12px' }} />
                <div style={{ fontWeight: 600, fontSize: '15px' }}>Processing Failed</div>
                <div style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>{error}</div>
              </div>
            ) : loading ? (
              <div style={{ textAlign: 'center', color: '#38bdf8' }}>
                <div style={{ width: '40px', height: '40px', border: '3px solid #334155', borderTopColor: '#38bdf8', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 16px' }} />
                <div style={{ fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>Processing Video Pipeline</div>
                <div style={{ fontSize: '13px', color: '#94a3b8', marginTop: '6px' }}>{progressText}</div>
              </div>
            ) : result ? (
              <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#34d399', fontSize: '15px', fontWeight: 600 }}>
                  <CheckCircle2 size={18} />
                  Batch Video Analysis Complete
                </div>
                <div style={{ fontSize: '13px', color: '#94a3b8' }}>
                  Analyzed video <strong>{selectedFile?.name}</strong> across {result.frames_processed} sampled frames.
                  Detected {result.total_unique_tracks} unique objects across {Object.keys(classCounts).length} distinct classes.
                </div>

                {/* Tracking Telemetry Table */}
                <div style={{ backgroundColor: '#1e293b', borderRadius: '8px', border: '1px solid #334155', padding: '16px', marginTop: '8px' }}>
                  <h4 style={{ margin: '0 0 12px', fontSize: '13px', color: '#f8fafc' }}>
                    Track Summary Log
                  </h4>
                  <TrackTable tracks={tracks.slice(0, 50)} />
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', color: '#64748b' }}>
                <Film size={48} style={{ margin: '0 auto 16px', opacity: 0.4 }} />
                <div style={{ fontSize: '16px', fontWeight: 600, color: '#94a3b8' }}>
                  Video Processing Workspace
                </div>
                <div style={{ fontSize: '13px', marginTop: '6px', maxWidth: '420px' }}>
                  Select a video file to run offline batch tracking, or switch to <strong>Live Detection</strong> for real-time WebSocket streaming playback.
                </div>
                <button
                  onClick={onSwitchToLiveStream}
                  style={{
                    marginTop: '16px',
                    padding: '8px 16px',
                    borderRadius: '6px',
                    border: '1px solid #334155',
                    backgroundColor: '#1e293b',
                    color: '#38bdf8',
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                >
                  Switch to Live Streaming Mode →
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Right Controls Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* File Upload Box */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
              Load Video File
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
              <span style={{ fontSize: '13px', fontWeight: 500, color: '#f8fafc' }}>
                {selectedFile ? selectedFile.name : 'Choose a video'}
              </span>
              <span style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>MP4, AVI, MOV, MKV up to 50MB</span>
              <input
                type="file"
                accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </label>
          </div>

          {/* Parameters */}
          <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
              Video Ingestion Parameters
            </h3>

            {/* Frame Stride Slider */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                <span style={{ color: '#94a3b8' }}>Frame Stride (Skip)</span>
                <span style={{ fontWeight: 600, color: '#38bdf8', fontFamily: 'monospace' }}>
                  Every {stride} {stride === 1 ? 'frame' : 'frames'}
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={stride}
                onChange={(e) => setStride(parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
              <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                Higher stride processes video faster with lower temporal granularity.
              </div>
            </div>

            {/* Confidence Slider */}
            <div style={{ marginBottom: '20px' }}>
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

            {/* Run Button */}
            <button
              onClick={handleProcessVideo}
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
              {loading ? 'Processing Video...' : 'Analyze Video Tracks'}
            </button>
          </div>

          {/* Class Breakdown if Result */}
          {result && (
            <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
              <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
                Detected Class Distribution
              </h3>
              <ClassDistribution classCounts={classCounts} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
