import React from 'react';

export function ClassDistribution({ classCounts = {} }) {
  const entries = Object.entries(classCounts || {}).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((acc, curr) => acc + curr[1], 0);

  if (entries.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '16px', color: '#64748b', fontSize: '13px' }}>
        No class distribution data available.
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {entries.map(([className, count]) => {
        const pct = total > 0 ? Math.round((count / total) * 100) : 0;
        return (
          <div key={className}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
              <span style={{ color: '#cbd5e1', textTransform: 'capitalize', fontWeight: 500 }}>
                {className}
              </span>
              <span style={{ color: '#94a3b8', fontFamily: 'monospace' }}>
                {count} ({pct}%)
              </span>
            </div>
            <div style={{ width: '100%', height: '6px', backgroundColor: '#334155', borderRadius: '3px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${pct}%`,
                  height: '100%',
                  backgroundColor: '#0284c7',
                  borderRadius: '3px',
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
