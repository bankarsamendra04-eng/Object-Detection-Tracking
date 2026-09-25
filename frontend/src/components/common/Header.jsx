import React from 'react';
import { StatusBadge } from './StatusBadge';
import { Cpu, Zap, RefreshCw, Layers } from 'lucide-react';

export function Header({ health, wsStatus, config, onRefresh, refreshing }) {
  const isHealthy = health?.status === 'healthy';
  const effectiveDevice = config?.device || health?.effective_device || 'Auto';
  const isGPU = effectiveDevice.toLowerCase().includes('cuda');
  const activeModel = config?.active_model_id || health?.model?.name || 'yolov8n.pt';

  return (
    <header
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '16px 24px',
        backgroundColor: '#0f172a',
        borderBottom: '1px solid #334155',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}
    >
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '28px',
              height: '28px',
              borderRadius: '6px',
              backgroundColor: '#0284c7',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Zap size={16} color="#fff" />
          </div>
          <h1 style={{ fontSize: '18px', fontWeight: 700, margin: 0, color: '#f8fafc', letterSpacing: '-0.02em' }}>
            VisionTrack AI
          </h1>
          <span style={{ fontSize: '11px', backgroundColor: '#1e293b', border: '1px solid #334155', color: '#38bdf8', padding: '2px 8px', borderRadius: '4px', fontWeight: 600 }}>
            v1.0.0
          </span>
        </div>
        <p style={{ margin: '2px 0 0 38px', fontSize: '12px', color: '#94a3b8' }}>
          Real-Time Object Detection, ByteTrack & Spatial Analytics
        </p>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
        {/* Device pill */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '4px 10px',
            borderRadius: '6px',
            backgroundColor: '#1e293b',
            border: '1px solid #334155',
            fontSize: '12px',
            color: isGPU ? '#38bdf8' : '#cbd5e1',
          }}
        >
          <Cpu size={14} color={isGPU ? '#38bdf8' : '#94a3b8'} />
          <span>Device: {effectiveDevice}</span>
        </div>

        {/* Active Model pill */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '4px 10px',
            borderRadius: '6px',
            backgroundColor: '#1e293b',
            border: '1px solid #334155',
            fontSize: '12px',
            color: '#a855f7',
          }}
        >
          <Layers size={14} color="#a855f7" />
          <span>Model: {activeModel}</span>
        </div>

        {/* Backend REST Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ fontSize: '12px', color: '#94a3b8' }}>API:</span>
          <StatusBadge status={isHealthy ? 'healthy' : 'offline'} label={isHealthy ? 'ONLINE' : 'OFFLINE'} />
        </div>

        {/* WebSocket Stream Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ fontSize: '12px', color: '#94a3b8' }}>Stream:</span>
          <StatusBadge status={wsStatus} />
        </div>

        {/* Refresh button */}
        <button
          onClick={onRefresh}
          disabled={refreshing}
          aria-label="Refresh System Status"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '32px',
            height: '32px',
            backgroundColor: '#1e293b',
            border: '1px solid #334155',
            borderRadius: '6px',
            color: '#94a3b8',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
        </button>
      </div>
    </header>
  );
}
