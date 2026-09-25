import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, RefreshCw, AlertCircle, ArrowDownRight, ArrowUpRight } from 'lucide-react';
import { MetricCard } from '../common/MetricCard';
import { ClassDistribution } from '../detection/ClassDistribution';
import { api } from '../../api/client';

export function AnalyticsDashboard({ streamFrame }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [resetting, setResetting] = useState(false);

  const fetchReport = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getAnalyticsReport();
      setReport(data);
    } catch (err) {
      setError(err.message || 'Failed to fetch analytics report');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReport();
    const timer = setInterval(fetchReport, 10000);
    return () => clearInterval(timer);
  }, []);

  const handleReset = async () => {
    try {
      setResetting(true);
      await api.resetAnalytics();
      await fetchReport();
    } catch (err) {
      setError(`Reset failed: ${err.message}`);
    } finally {
      setResetting(false);
    }
  };

  // Merge report data or live stream frame analytics
  const streamAnalytics = streamFrame?.analytics || {};
  const totalUnique = streamAnalytics.total_unique_objects ?? report?.overall_summary?.total_unique_objects ?? 0;
  const activeObjects = streamAnalytics.active_objects ?? report?.overall_summary?.active_objects ?? 0;
  const hasStreamCrossings = streamAnalytics.line_crossings_in !== undefined || streamAnalytics.line_crossings_out !== undefined;
  const totalCrossings = hasStreamCrossings
    ? (streamAnalytics.line_crossings_in || 0) + (streamAnalytics.line_crossings_out || 0)
    : (report?.line_crossing?.total_crossings ?? 0);
  const crossingsIn = streamAnalytics.line_crossings_in ?? report?.line_crossing?.direction_breakdown?.INBOUND ?? 0;
  const crossingsOut = streamAnalytics.line_crossings_out ?? report?.line_crossing?.direction_breakdown?.OUTBOUND ?? 0;
  const avgFps = streamAnalytics.fps ?? report?.performance?.average_fps ?? 0;
  const classStats = streamAnalytics.class_statistics ?? report?.class_statistics ?? {};

  // Normalize class counts
  const classCounts = {};
  Object.entries(classStats).forEach(([cls, stat]) => {
    classCounts[cls] = typeof stat === 'number' ? stat : (stat.unique_count || stat.count || 0);
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Controls Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 700, margin: 0, color: '#f8fafc' }}>
            Spatial Vision & Telemetry Analytics
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#94a3b8' }}>
            Persistent ID tracking, dwell times, and virtual tripwire crossings
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={fetchReport}
            disabled={loading}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 12px',
              borderRadius: '6px',
              border: '1px solid #334155',
              backgroundColor: '#1e293b',
              color: '#e2e8f0',
              fontSize: '12px',
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>

          <button
            onClick={handleReset}
            disabled={resetting}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 12px',
              borderRadius: '6px',
              border: '1px solid #ef4444',
              backgroundColor: 'rgba(239, 68, 68, 0.15)',
              color: '#f87171',
              fontSize: '12px',
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            {resetting ? 'Resetting...' : 'Reset Session'}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: '12px 16px', backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: '8px', color: '#fca5a5', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Primary KPI Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        <MetricCard
          title="Total Unique Objects"
          value={totalUnique}
          subtitle="Cumulative tracked entities"
          icon={TrendingUp}
          color="#38bdf8"
        />
        <MetricCard
          title="Active Entities Now"
          value={activeObjects}
          subtitle="In current field of view"
          icon={BarChart3}
          color="#4ade80"
        />
        <MetricCard
          title="Inbound Crossings"
          value={crossingsIn}
          subtitle="Tripwire entered"
          icon={ArrowDownRight}
          color="#10b981"
        />
        <MetricCard
          title="Outbound Crossings"
          value={crossingsOut}
          subtitle="Tripwire exited"
          icon={ArrowUpRight}
          color="#f43f5e"
        />
        <MetricCard
          title="System FPS"
          value={typeof avgFps === 'number' ? avgFps.toFixed(1) : avgFps}
          unit="fps"
          subtitle="End-to-end throughput"
          color="#f59e0b"
        />
      </div>

      {/* Main Analytics Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 340px', gap: '20px' }}>
        {/* Left: Event Logs & Crossing Summary */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
            <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
              Virtual Tripwire Telemetry
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
              <div style={{ backgroundColor: '#0f172a', padding: '16px', borderRadius: '8px', border: '1px solid #334155' }}>
                <div style={{ fontSize: '12px', color: '#94a3b8' }}>Configured Virtual Tripwire</div>
                <div style={{ fontSize: '13px', fontFamily: 'monospace', color: '#38bdf8', marginTop: '6px' }}>
                  (0, 360) ➔ (1280, 360)
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>Horizontal midline tripwire</div>
              </div>

              <div style={{ backgroundColor: '#0f172a', padding: '16px', borderRadius: '8px', border: '1px solid #334155' }}>
                <div style={{ fontSize: '12px', color: '#94a3b8' }}>Total Tripwire Crossings</div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>
                  {totalCrossings}
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>Anti-duplicate debounced</div>
              </div>
            </div>

            {/* Recent Events List */}
            <h4 style={{ margin: '0 0 12px', fontSize: '13px', fontWeight: 600, color: '#cbd5e1' }}>
              Recent Telemetry Event Log
            </h4>

            {report?.recent_events && report.recent_events.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '280px', overflowY: 'auto' }}>
                {report.recent_events.slice(0, 20).map((evt, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      backgroundColor: '#0f172a',
                      border: '1px solid #334155',
                      fontSize: '12px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span
                        style={{
                          padding: '2px 6px',
                          borderRadius: '4px',
                          fontSize: '10px',
                          fontWeight: 700,
                          backgroundColor:
                            evt.event_type === 'LINE_CROSSED' ? '#0284c7' :
                            evt.event_type === 'OBJECT_ENTERED' ? '#059669' : '#dc2626',
                          color: '#fff',
                        }}
                      >
                        {evt.event_type}
                      </span>
                      <span style={{ color: '#e2e8f0', fontWeight: 500 }}>
                        Track #{evt.track_id} ({evt.class_name})
                      </span>
                    </div>
                    <span style={{ color: '#64748b', fontFamily: 'monospace', fontSize: '11px' }}>
                      {evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : '--'}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '24px', color: '#64748b', fontSize: '13px' }}>
                No events recorded yet. Start a live stream or analyze a video to accumulate telemetry events.
              </div>
            )}
          </div>
        </div>

        {/* Right: Class Distribution */}
        <div style={{ backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', padding: '20px' }}>
          <h3 style={{ margin: '0 0 16px', fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
            Class Demographics
          </h3>
          <ClassDistribution classCounts={classCounts} />
        </div>
      </div>
    </div>
  );
}
