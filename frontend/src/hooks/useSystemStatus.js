import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';

export function useSystemStatus() {
  const [health, setHealth] = useState(null);
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStatus = useCallback(async () => {
    try {
      const [healthData, configData] = await Promise.all([
        api.getHealth().catch((err) => ({ status: 'offline', error: err.message })),
        api.getConfig().catch(() => null),
      ]);
      setHealth(healthData);
      setConfig(configData);
      setError(healthData.status === 'offline' ? healthData.error : null);
    } catch (err) {
      setError(err.message || 'Failed connecting to backend.');
      setHealth({ status: 'offline' });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  return {
    health,
    config,
    loading,
    error,
    refresh: fetchStatus,
    isHealthy: health?.status === 'healthy',
  };
}
