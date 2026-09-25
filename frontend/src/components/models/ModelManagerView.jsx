import React, { useState, useEffect } from 'react';
import { Database, CheckCircle2, AlertCircle, RefreshCw, Cpu, Layers, ShieldCheck, ArrowRight } from 'lucide-react';
import { StatusBadge } from '../common/StatusBadge';
import { Modal } from '../common/Modal';
import { api } from '../../api/client';

export function ModelManagerView({ onModelSwitched }) {
  const [modelsData, setModelsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [actionLoading, setActionLoading] = useState(null);
  const [validationResult, setValidationResult] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);

  const fetchModels = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.listModels();
      setModelsData(data);
    } catch (err) {
      setError(err.message || 'Failed to fetch registered models');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleValidate = async (modelId) => {
    try {
      setActionLoading(`val_${modelId}`);
      const res = await api.validateModel(modelId);
      setValidationResult(res);
      setModalOpen(true);
    } catch (err) {
      setError(`Validation failed: ${err.message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleSwitch = async (modelId) => {
    try {
      setActionLoading(`sw_${modelId}`);
      const res = await api.switchModel(modelId);
      await fetchModels();
      if (onModelSwitched) onModelSwitched(res);
    } catch (err) {
      setError(`Model switch failed: ${err.message}`);
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 700, margin: 0, color: '#f8fafc' }}>
            Model Registry & Checkpoint Manager
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#94a3b8' }}>
            Inspect, validate, and switch between general pretrained and domain-adapted custom models
          </p>
        </div>

        <button
          onClick={fetchModels}
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
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Refresh Registry
        </button>
      </div>

      {error && (
        <div style={{ padding: '12px 16px', backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: '8px', color: '#fca5a5', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Model Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '20px' }}>
        {modelsData?.models?.map((model) => {
          const isActive = model.is_active || model.model_id === modelsData.active_model_id;
          const isPretrained = model.model_type === 'pretrained';
          const isAvailable = model.status === 'AVAILABLE' || model.status === 'LOADED';

          return (
            <div
              key={model.model_id}
              style={{
                backgroundColor: '#1e293b',
                borderRadius: '12px',
                border: isActive ? '2px solid #0284c7' : '1px solid #334155',
                padding: '20px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                position: 'relative',
              }}
            >
              {isActive && (
                <div
                  style={{
                    position: 'absolute',
                    top: '-10px',
                    right: '16px',
                    backgroundColor: '#0284c7',
                    color: '#fff',
                    padding: '2px 10px',
                    borderRadius: '12px',
                    fontSize: '11px',
                    fontWeight: 700,
                    letterSpacing: '0.04em',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                  }}
                >
                  ACTIVE INFERENCE MODEL
                </div>
              )}

              <div>
                {/* Title & Badge */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 600, color: '#f8fafc' }}>
                      {model.model_name}
                    </h3>
                    <div style={{ fontSize: '12px', color: '#64748b', fontFamily: 'monospace', marginTop: '2px' }}>
                      ID: {model.model_id} • v{model.version}
                    </div>
                  </div>
                  <StatusBadge status={model.status} />
                </div>

                {/* Badges row */}
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px' }}>
                  <span
                    style={{
                      padding: '2px 8px',
                      borderRadius: '4px',
                      backgroundColor: isPretrained ? 'rgba(56, 189, 248, 0.15)' : 'rgba(168, 85, 247, 0.15)',
                      color: isPretrained ? '#38bdf8' : '#c084fc',
                      fontSize: '11px',
                      fontWeight: 600,
                      border: `1px solid ${isPretrained ? 'rgba(56, 189, 248, 0.3)' : 'rgba(168, 85, 247, 0.3)'}`,
                    }}
                  >
                    {isPretrained ? 'Pretrained Checkpoint' : 'Custom Trained'}
                  </span>

                  <span
                    style={{
                      padding: '2px 8px',
                      borderRadius: '4px',
                      backgroundColor: '#0f172a',
                      color: '#94a3b8',
                      fontSize: '11px',
                      border: '1px solid #334155',
                      fontFamily: 'monospace',
                    }}
                  >
                    Input: {model.input_size}×{model.input_size}px
                  </span>

                  <span
                    style={{
                      padding: '2px 8px',
                      borderRadius: '4px',
                      backgroundColor: '#0f172a',
                      color: '#94a3b8',
                      fontSize: '11px',
                      border: '1px solid #334155',
                    }}
                  >
                    {model.num_classes} Classes
                  </span>
                </div>

                {/* Details specs */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '12px', marginBottom: '16px', backgroundColor: '#0f172a', padding: '12px', borderRadius: '8px', border: '1px solid #334155' }}>
                  <div>
                    <span style={{ color: '#64748b' }}>Weights File: </span>
                    <span style={{ color: '#cbd5e1', fontFamily: 'monospace' }}>{model.weights_path}</span>
                  </div>
                  <div>
                    <span style={{ color: '#64748b' }}>Framework: </span>
                    <span style={{ color: '#cbd5e1' }}>{model.framework}</span>
                  </div>
                  <div>
                    <span style={{ color: '#64748b' }}>Rec. Confidence: </span>
                    <span style={{ color: '#38bdf8', fontFamily: 'monospace' }}>{model.recommended_confidence}</span>
                  </div>
                  <div>
                    <span style={{ color: '#64748b' }}>Rec. IoU: </span>
                    <span style={{ color: '#38bdf8', fontFamily: 'monospace' }}>{model.recommended_iou}</span>
                  </div>
                </div>

                {/* Notes */}
                {model.notes && (
                  <p style={{ margin: '0 0 16px', fontSize: '12px', color: '#94a3b8', lineHeight: 1.5 }}>
                    {model.notes}
                  </p>
                )}

                {/* Class Tags Preview */}
                {model.classes && Object.keys(model.classes).length > 0 && (
                  <div style={{ marginBottom: '16px' }}>
                    <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '6px', fontWeight: 600 }}>
                      Class Schema ({Object.keys(model.classes).length} categories):
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', maxHeight: '72px', overflowY: 'auto' }}>
                      {Object.entries(model.classes).map(([id, name]) => (
                        <span
                          key={id}
                          style={{
                            padding: '2px 6px',
                            borderRadius: '4px',
                            backgroundColor: '#0f172a',
                            border: '1px solid #334155',
                            fontSize: '11px',
                            color: '#e2e8f0',
                          }}
                        >
                          {name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div style={{ display: 'flex', gap: '8px', borderTop: '1px solid #334155', paddingTop: '16px' }}>
                <button
                  onClick={() => handleValidate(model.model_id)}
                  disabled={actionLoading !== null}
                  style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    padding: '8px 14px',
                    borderRadius: '6px',
                    border: '1px solid #334155',
                    backgroundColor: '#0f172a',
                    color: '#e2e8f0',
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: actionLoading !== null ? 'not-allowed' : 'pointer',
                  }}
                >
                  <ShieldCheck size={15} color="#38bdf8" />
                  {actionLoading === `val_${model.model_id}` ? 'Validating...' : 'Validate Weights'}
                </button>

                <button
                  onClick={() => handleSwitch(model.model_id)}
                  disabled={isActive || !isAvailable || actionLoading !== null}
                  style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    padding: '8px 14px',
                    borderRadius: '6px',
                    border: 'none',
                    backgroundColor: isActive ? '#059669' : (!isAvailable ? '#334155' : '#0284c7'),
                    color: '#fff',
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: isActive || !isAvailable || actionLoading !== null ? 'not-allowed' : 'pointer',
                    opacity: !isAvailable ? 0.6 : 1,
                  }}
                >
                  {isActive ? (
                    <>
                      <CheckCircle2 size={15} /> Active
                    </>
                  ) : !isAvailable ? (
                    'Not Available'
                  ) : actionLoading === `sw_${model.model_id}` ? (
                    'Switching...'
                  ) : (
                    <>
                      Activate <ArrowRight size={14} />
                    </>
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Validation Results Modal */}
      <Modal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        title={`Model Validation Report: ${validationResult?.model_id}`}
      >
        {validationResult && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ color: '#94a3b8' }}>Validation Status:</span>
              <StatusBadge status={validationResult.status} />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Weights Available on Disk:</span>
              <span style={{ fontWeight: 600, color: validationResult.available ? '#34d399' : '#f87171' }}>
                {validationResult.available ? 'YES (Present)' : 'NO (Missing)'}
              </span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Loadable by Ultralytics:</span>
              <span style={{ fontWeight: 600, color: validationResult.loadable ? '#34d399' : '#f87171' }}>
                {validationResult.loadable ? 'YES (Verified)' : 'NO'}
              </span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Verified Device:</span>
              <span style={{ color: '#38bdf8', fontFamily: 'monospace' }}>{validationResult.device}</span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Supported Classes:</span>
              <span style={{ color: '#f8fafc', fontWeight: 600 }}>{validationResult.num_classes}</span>
            </div>

            {validationResult.test_inference_time_ms && (
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Test Forward Pass Latency:</span>
                <span style={{ color: '#f59e0b', fontFamily: 'monospace', fontWeight: 600 }}>
                  {validationResult.test_inference_time_ms} ms
                </span>
              </div>
            )}

            {validationResult.error && (
              <div style={{ padding: '10px 12px', backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: '6px', color: '#fca5a5' }}>
                <strong>Error: </strong> {validationResult.error}
              </div>
            )}

            {validationResult.notes && (
              <div style={{ color: '#94a3b8', fontSize: '12px', borderTop: '1px solid #334155', paddingTop: '10px' }}>
                {validationResult.notes}
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
