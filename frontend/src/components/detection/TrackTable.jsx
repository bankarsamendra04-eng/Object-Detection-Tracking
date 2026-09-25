import React from 'react';

export function TrackTable({ tracks = [] }) {
  if (!tracks || tracks.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '24px', color: '#64748b', fontSize: '13px' }}>
        No active tracked entities in frame.
      </div>
    );
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
            <th style={{ padding: '8px 12px', fontWeight: 600 }}>ID</th>
            <th style={{ padding: '8px 12px', fontWeight: 600 }}>Class</th>
            <th style={{ padding: '8px 12px', fontWeight: 600 }}>Confidence</th>
            <th style={{ padding: '8px 12px', fontWeight: 600 }}>Center (X, Y)</th>
            <th style={{ padding: '8px 12px', fontWeight: 600 }}>Size (W × H)</th>
          </tr>
        </thead>
        <tbody>
          {tracks.map((track) => {
            const box = track.box || track;
            const w = Math.round(box.x2 - box.x1);
            const h = Math.round(box.y2 - box.y1);
            const cx = Math.round((box.x1 + box.x2) / 2);
            const cy = Math.round((box.y1 + box.y2) / 2);
            const conf = Math.round((track.confidence || 0) * 100);

            return (
              <tr
                key={track.track_id}
                style={{
                  borderBottom: '1px solid #1e293b',
                  color: '#e2e8f0',
                  transition: 'background-color 0.15s',
                }}
              >
                <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontWeight: 700, color: '#38bdf8' }}>
                  #{track.track_id}
                </td>
                <td style={{ padding: '8px 12px', fontWeight: 500, textTransform: 'capitalize' }}>
                  {track.class_name}
                </td>
                <td style={{ padding: '8px 12px', fontFamily: 'monospace' }}>
                  {conf}%
                </td>
                <td style={{ padding: '8px 12px', fontFamily: 'monospace', color: '#94a3b8' }}>
                  ({cx}, {cy})
                </td>
                <td style={{ padding: '8px 12px', fontFamily: 'monospace', color: '#94a3b8' }}>
                  {w} × {h} px
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
