import React from 'react';

export function StatusBadge({ status, label = null }) {
  const normalized = (status || 'unknown').toLowerCase();

  const styles = {
    healthy: { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399', dot: '#10b981', border: 'rgba(16, 185, 129, 0.3)' },
    connected: { bg: 'rgba(56, 189, 248, 0.15)', text: '#38bdf8', dot: '#0284c7', border: 'rgba(56, 189, 248, 0.3)' },
    streaming: { bg: 'rgba(168, 85, 247, 0.15)', text: '#c084fc', dot: '#a855f7', border: 'rgba(168, 85, 247, 0.3)' },
    paused: { bg: 'rgba(245, 158, 11, 0.15)', text: '#fbbf24', dot: '#f59e0b', border: 'rgba(245, 158, 11, 0.3)' },
    offline: { bg: 'rgba(239, 68, 68, 0.15)', text: '#f87171', dot: '#ef4444', border: 'rgba(239, 68, 68, 0.3)' },
    error: { bg: 'rgba(239, 68, 68, 0.15)', text: '#f87171', dot: '#ef4444', border: 'rgba(239, 68, 68, 0.3)' },
    disconnected: { bg: 'rgba(148, 163, 184, 0.15)', text: '#94a3b8', dot: '#64748b', border: 'rgba(148, 163, 184, 0.3)' },
    available: { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399', dot: '#10b981', border: 'rgba(16, 185, 129, 0.3)' },
    not_available: { bg: 'rgba(148, 163, 184, 0.15)', text: '#94a3b8', dot: '#64748b', border: 'rgba(148, 163, 184, 0.3)' },
    loaded: { bg: 'rgba(56, 189, 248, 0.15)', text: '#38bdf8', dot: '#0284c7', border: 'rgba(56, 189, 248, 0.3)' },
    valid: { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399', dot: '#10b981', border: 'rgba(16, 185, 129, 0.3)' },
    invalid: { bg: 'rgba(239, 68, 68, 0.15)', text: '#f87171', dot: '#ef4444', border: 'rgba(239, 68, 68, 0.3)' },
  };

  const style = styles[normalized] || styles.disconnected;
  const displayLabel = label || status?.toUpperCase() || 'UNKNOWN';

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: '3px 10px',
        borderRadius: '9999px',
        fontSize: '11px',
        fontWeight: 600,
        letterSpacing: '0.04em',
        backgroundColor: style.bg,
        color: style.text,
        border: `1px solid ${style.border}`,
      }}
    >
      <span
        style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          backgroundColor: style.dot,
        }}
      />
      {displayLabel}
    </span>
  );
}
