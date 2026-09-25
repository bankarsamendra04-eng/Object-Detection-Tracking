import React from 'react';

export function MetricCard({ title, value, unit = '', subtitle = null, icon: Icon, color = '#38bdf8' }) {
  return (
    <div
      style={{
        backgroundColor: '#1e293b',
        borderRadius: '10px',
        padding: '16px',
        border: '1px solid #334155',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        minWidth: 0,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <span style={{ fontSize: '12px', fontWeight: 500, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          {title}
        </span>
        {Icon && <Icon size={16} color={color} />}
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
        <span style={{ fontSize: '24px', fontWeight: 700, color: color, fontFamily: 'monospace' }}>
          {value !== null && value !== undefined ? value : 'N/A'}
        </span>
        {unit && <span style={{ fontSize: '12px', color: '#64748b' }}>{unit}</span>}
      </div>

      {subtitle && (
        <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
          {subtitle}
        </div>
      )}
    </div>
  );
}
