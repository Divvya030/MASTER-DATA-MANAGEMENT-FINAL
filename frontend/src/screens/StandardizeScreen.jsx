import React, { useState } from 'react';
import { api } from '../api';

export default function StandardizeScreen({ dataset, onStandardized }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      const r = await api.runStandardize(dataset.dataset_id);
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-header">
        <h1>Standardize &amp; cleanse</h1>
        <p>Transformations are driven entirely by the cleansing signals found during profiling. The raw layer is never touched — this writes a new STANDARDIZED layer.</p>
      </div>

      {error && <div className="status-banner error">{error}</div>}

      {!result && (
        <div className="panel">
          <button className="btn btn-primary" disabled={loading} onClick={run}>
            {loading ? 'Standardizing…' : 'Run standardization'}
          </button>
        </div>
      )}

      {result && (
        <>
          <div className="panel">
            <div className="panel-title">Transformations applied</div>
            <div className="metric-row" style={{ flexWrap: 'wrap' }}>
              {Object.entries(result.transformation_summary).map(([t, count]) => (
                <div className="metric" key={t}><b>{count.toLocaleString()}</b> × {t.replace(/_/g, ' ').toLowerCase()}</div>
              ))}
              {Object.keys(result.transformation_summary).length === 0 && (
                <div className="metric">No transformations were needed — this field set was already clean.</div>
              )}
            </div>
            <div className="metric" style={{ marginTop: 10 }}>total transformations: <b>{result.transformation_count.toLocaleString()}</b></div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button className="btn btn-primary" onClick={onStandardized}>Continue to validation →</button>
          </div>
        </>
      )}
    </div>
  );
}
