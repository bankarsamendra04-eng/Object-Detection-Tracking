import React from 'react';
import { Camera, Image, Film, BarChart3, Database, Sliders } from 'lucide-react';

export function Sidebar({ activeTab, onSelectTab }) {
  const navItems = [
    { id: 'live', label: 'Live Detection', icon: Camera, desc: 'Real-time WebSocket' },
    { id: 'image', label: 'Image Inference', icon: Image, desc: 'REST single frame' },
    { id: 'video', label: 'Video Analysis', icon: Film, desc: 'Batch file tracking' },
    { id: 'analytics', label: 'Spatial Analytics', icon: BarChart3, desc: 'Crossings & stats' },
    { id: 'models', label: 'Model Registry', icon: Database, desc: 'Pretrained & Custom' },
    { id: 'settings', label: 'Configuration', icon: Sliders, desc: 'Thresholds & probe' },
  ];

  return (
    <aside
      style={{
        width: '260px',
        backgroundColor: '#0f172a',
        borderRight: '1px solid #334155',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        padding: '20px 14px',
        flexShrink: 0,
      }}
    >
      <div>
        <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', padding: '0 12px 12px' }}>
          Navigation
        </div>
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                aria-current={isActive ? 'page' : undefined}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '10px 14px',
                  borderRadius: '8px',
                  border: 'none',
                  backgroundColor: isActive ? '#0284c7' : 'transparent',
                  color: isActive ? '#ffffff' : '#cbd5e1',
                  cursor: 'pointer',
                  textAlign: 'left',
                  width: '100%',
                  transition: 'all 0.15s ease-in-out',
                }}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = '#1e293b';
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <Icon size={18} color={isActive ? '#ffffff' : '#94a3b8'} />
                <div>
                  <div style={{ fontSize: '13px', fontWeight: 600 }}>{item.label}</div>
                  <div style={{ fontSize: '11px', color: isActive ? '#bae6fd' : '#64748b' }}>{item.desc}</div>
                </div>
              </button>
            );
          })}
        </nav>
      </div>

      <div
        style={{
          backgroundColor: '#1e293b',
          borderRadius: '8px',
          padding: '12px',
          border: '1px solid #334155',
          fontSize: '11px',
          color: '#94a3b8',
        }}
      >
        <div style={{ fontWeight: 600, color: '#e2e8f0', marginBottom: '4px' }}>Hardware Engine</div>
        <div>Inference: Ultralytics YOLOv8</div>
        <div>Tracker: ByteTrack Persistent</div>
        <div style={{ marginTop: '6px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#10b981' }} />
          Local Offline Runtime
        </div>
      </div>
    </aside>
  );
}
