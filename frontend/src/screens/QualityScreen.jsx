import React, { useState, useEffect } from 'react';
import { api } from '../api';

const STATUS_BADGE = {
  FIXED: 'badge-good',
};

export default function QualityScreen({ dataset, onValidated }) {
  const [auditReport, setAuditReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.qualityComparison(dataset.dataset_id)
      .then(r => setAuditReport(r.signal_audit_report || []))
      .catch(() => setAuditReport(null));
  }, [dataset.dataset_id]);

  // The report shows only what standardization actually FIXED. Entries with another status
  // (PARTIALLY_FIXED, PRESERVED, NOT_APPLICABLE) are still returned by the API and kept in the data,
  // they are just not listed on this screen.
  const fixedOnly = (auditReport || []).filter(e => e.status === 'FIXED');

  async function run() {
    setLoading(true);
    setError(null);
    try {
      await api.runReprofile(dataset.dataset_id);
      const r = await api.qualityComparison(dataset.dataset_id);
      setAuditReport(r.signal_audit_report || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-header">
        <h1>Standardization audit report</h1>
        <p>The same profiling engine re-runs on the STANDARDIZED layer, then this compares it against RAW to list the issues standardization fixed automatically. Standardization ran with no approval step; this page only reports what happened.</p>
      </div>

      {error && <div className="status-banner error">{error}</div>}

      {!auditReport && (
        <div className="panel">
          <button className="btn btn-primary" disabled={loading} onClick={run}>
            {loading ? 'Re-profiling…' : 'Run re-profiling'}
          </button>
        </div>
      )}

      {auditReport && fixedOnly.length === 0 && (
        <div className="panel">
          <div className="metric">
            {auditReport.length === 0
              ? 'No cleansing signals were detected on the RAW data for this dataset — nothing to report.'
              : 'Standardization did not fully fix any signal on this dataset — nothing to list.'}
          </div>
        </div>
      )}

      {auditReport && fixedOnly.length > 0 && (
        <div className="panel">
          <div className="panel-title">Fixed by standardization ({fixedOnly.length})</div>
          {fixedOnly.map((e, i) => (
            <div className="field-card" key={i}>
              <div className="field-card-head">
                <span className="field-name">{e.field} · {e.signal}</span>
                <span className={`badge ${STATUS_BADGE[e.status] || 'badge-neutral'}`}>{e.status.replace(/_/g, ' ')}</span>
              </div>
              <div className="metric-row">
                <div className="metric">fixed <b>{e.fixed_count.toLocaleString()}</b>{e.is_estimated ? ' (estimated)' : ''}</div>
                <div className="metric">remaining <b>{e.remaining_count.toLocaleString()}</b>{e.is_estimated ? ' (estimated)' : ''}</div>
              </div>
              <div className="metric" style={{ marginTop: 6, color: 'var(--slate)' }}>{e.explanation}</div>
            </div>
          ))}
        </div>
      )}

      {auditReport && (
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn btn-primary" onClick={onValidated}>Continue to ER configuration →</button>
        </div>
      )}
    </div>
  );
}
