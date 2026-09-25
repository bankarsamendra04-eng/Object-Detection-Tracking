import React, { useState } from 'react';
import { Header } from './components/common/Header';
import { Sidebar } from './components/common/Sidebar';
import { LiveDetection } from './components/detection/LiveDetection';
import { ImageDetection } from './components/detection/ImageDetection';
import { VideoDetection } from './components/detection/VideoDetection';
import { AnalyticsDashboard } from './components/analytics/AnalyticsDashboard';
import { ModelManagerView } from './components/models/ModelManagerView';
import { SettingsView } from './components/settings/SettingsView';
import { useSystemStatus } from './hooks/useSystemStatus';
import { useWebSocketStream } from './hooks/useWebSocketStream';

export default function App() {
  const [activeTab, setActiveTab] = useState('live');
  const { health, config, loading, error, refresh, isHealthy } = useSystemStatus();
  const streamState = useWebSocketStream();

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#0b1120', color: '#f8fafc' }}>
      {/* Top Header */}
      <Header
        health={health}
        wsStatus={streamState.status}
        config={config}
        onRefresh={refresh}
        refreshing={loading}
      />

      {/* Main Body with Sidebar and Active Workspace */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        {/* Sidebar Navigation */}
        <Sidebar activeTab={activeTab} onSelectTab={setActiveTab} />

        {/* Dynamic Viewport Container */}
        <main style={{ flex: 1, padding: '24px', overflowY: 'auto', backgroundColor: '#0b1120' }}>
          {!isHealthy && (
            <div
              style={{
                marginBottom: '20px',
                padding: '12px 16px',
                borderRadius: '8px',
                backgroundColor: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid #ef4444',
                color: '#fca5a5',
                fontSize: '13px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div>
                <strong>Backend Service Offline: </strong>
                Ensure the FastAPI backend server is running at <code>http://127.0.0.1:8000</code>.
              </div>
              <button
                onClick={refresh}
                style={{
                  padding: '4px 10px',
                  borderRadius: '4px',
                  border: '1px solid #ef4444',
                  backgroundColor: '#ef4444',
                  color: '#fff',
                  fontSize: '12px',
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
              >
                Retry Connection
              </button>
            </div>
          )}

          {activeTab === 'live' && (
            <LiveDetection
              streamState={streamState}
              config={config}
              onStart={streamState.start}
              onStop={streamState.stop}
              onPause={streamState.pause}
              onResume={streamState.resume}
            />
          )}

          {activeTab === 'image' && (
            <ImageDetection config={config} />
          )}

          {activeTab === 'video' && (
            <VideoDetection
              config={config}
              onSwitchToLiveStream={() => setActiveTab('live')}
            />
          )}

          {activeTab === 'analytics' && (
            <AnalyticsDashboard streamFrame={streamState.currentFrame} />
          )}

          {activeTab === 'models' && (
            <ModelManagerView onModelSwitched={refresh} />
          )}

          {activeTab === 'settings' && (
            <SettingsView config={config} />
          )}
        </main>
      </div>
    </div>
  );
}
