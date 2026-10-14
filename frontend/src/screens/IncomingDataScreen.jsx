import React, { useEffect, useState, useCallback } from 'react';
import { api } from '../api';

// Single record and a batch use the SAME service (POST /api/domains/{domain}/incoming
// accepts one-or-more records); a CSV/XLSX batch goes through the identical
// ingestion -> mapping -> standardization -> matching pipeline via /incoming/file.
export default function IncomingDataScreen({ domain }) {
  const [mode, setMode] = useState('form');   // form | file
  const [rows, setRows] = useState([{}]);
  const [schema, setSchema] = useState(null);         // canonical fields configured for this domain
  const [schemaError, setSchemaError] = useState(null);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [batches, setBatches] = useState([]);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [activeModel, setActiveModel] = useState(undefined); // undefined = checking

  // Incoming data REQUIRES an ACTIVE model (backend enforces it; pre-check
  // gives the user a friendly gate instead of a submission error).
  useEffect(() => {
    let cancelled = false;
    api.activeModel(domain)
      .then(m => { if (!cancelled) setActiveModel(m); })
      .catch(() => { if (!cancelled) setActiveModel(null); });
    return () => { cancelled = true; };
  }, [domain]);

  // The form shows EVERY canonical field configured for the domain (not a
  // fixed handful), and resets when the domain changes.
  useEffect(() => {
    let cancelled = false;
    setSchema(null); setSchemaError(null); setRows([{}]); setResult(null);
    api.domainSchema(domain)
      .then(r => { if (!cancelled) setSchema(r.fields); })
      .catch(e => { if (!cancelled) setSchemaError(e.message); });
    return () => { cancelled = true; };
  }, [domain]);

  const refresh = useCallback(() => {
    api.listIncomingBatches(domain).then(r => setBatches(r.batches)).catch(() => {});
  }, [domain]);

  useEffect(() => { refresh(); }, [refresh]);

  function updateRow(i, key, value) {
    setRows(prev => prev.map((r, idx) => idx === i ? { ...r, [key]: value } : r));
  }

  function removeRow(i) {
    setRows(prev => prev.length > 1 ? prev.filter((_, idx) => idx !== i) : [{}]);
  }

  async function submitForm() {
    setBusy(true); setError(null); setResult(null);
    try {
      const cleaned = rows
        .map(r => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === 'string' ? v.trim() : v]).filter(([, v]) => v)))
        .filter(r => Object.keys(r).length);
      if (!cleaned.length) { setError('Add at least one field to at least one record.'); setBusy(false); return; }
      const r = await api.submitIncomingRecords(domain, cleaned);
      setResult(r);
      refresh();
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  async function submitFile() {
    if (!file) { setError('Choose a CSV or XLSX file first.'); return; }
    setBusy(true); setError(null); setResult(null);
    try {
      const r = await api.submitIncomingFile(domain, file);
      setResult(r);
      refresh();
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  return (
    <div>
      <div className="page-header">
        <h1>Continuous Incoming Data</h1>
        <p>New records go through the same ingestion, mapping and standardization used for historical data, then
          are matched against the domain's ACTIVE model and mastered entities. This never trains or retrains a model.</p>
      </div>
      {error && <div className="status-banner error">{error}</div>}

      {activeModel === undefined && <div className="spinner-text">Checking for an ACTIVE model…</div>}
      {activeModel === null && (
        <div className="panel">
          <div className="panel-title">No ACTIVE model</div>
          <div className="evidence-drawer">
            Incoming records are matched against the domain's <b>ACTIVE</b> model. No model is currently
            active for <b>{domain}</b> — train, validate and activate a model first (see the dataset
            pipeline's Train stage).
          </div>
        </div>
      )}
      {activeModel != null && (
        <div className="evidence-drawer" style={{ marginBottom: 14 }}>
          Using ACTIVE model <b>{activeModel.training_run_id}</b> · incoming data never triggers retraining.
        </div>
      )}

      <div className="tabs">
        <div className={`tab ${mode === 'form' ? 'active' : ''}`} onClick={() => setMode('form')}>Single / manual records</div>
        <div className={`tab ${mode === 'file' ? 'active' : ''}`} onClick={() => setMode('file')}>Batch file (CSV/XLSX)</div>
      </div>

      {mode === 'form' ? (
        <div className="panel">
          <div className="panel-title">Records</div>
          {schemaError && <div className="status-banner error">Could not load the {domain} field list: {schemaError}</div>}
          {!schema && !schemaError && <div className="spinner-text">Loading {domain} fields…</div>}
          {schema && rows.map((row, i) => (
            <div key={i} className="evidence-drawer" style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <b>Record {i + 1}</b>
                {rows.length > 1 && <button className="btn btn-sm" onClick={() => removeRow(i)}>Remove</button>}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 10 }}>
                {schema.map(f => (
                  <label key={f.name} style={{ display: 'flex', flexDirection: 'column', gap: 3, fontSize: 12 }}>
                    <span style={{ color: 'var(--slate)' }}>
                      {f.label}{f.identifier && <span title="Identifier field" style={{ marginLeft: 4 }}>(ID)</span>}
                    </span>
                    <input
                      type={f.input_type || 'text'}
                      name={f.name}
                      aria-label={f.label}
                      value={row[f.name] || ''}
                      onChange={e => updateRow(i, f.name, e.target.value)}
                      style={{ padding: 6, fontSize: 12 }} />
                  </label>
                ))}
              </div>
            </div>
          ))}
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-sm" disabled={!schema} onClick={() => setRows(prev => [...prev, {}])}>+ Add record</button>
            <button className="btn btn-primary" disabled={busy || !schema || activeModel == null} onClick={submitForm}>{busy ? 'Submitting…' : 'Submit'}</button>
          </div>
          <div className="evidence-drawer" style={{ marginTop: 10, fontSize: 11 }}>
            These are the canonical fields configured for the <b>{domain}</b> domain. Fill in whatever you have;
            empty fields are left out and the record is matched on what is provided.
          </div>
        </div>
      ) : (
        <div className="panel">
          <div className="panel-title">Upload a batch file</div>
          <input type="file" accept=".csv,.xlsx" onChange={e => setFile(e.target.files?.[0] || null)} />
          <div style={{ marginTop: 12 }}>
            <button className="btn btn-primary" disabled={busy || activeModel == null} onClick={submitFile}>{busy ? 'Processing…' : 'Submit batch'}</button>
          </div>
        </div>
      )}

      {result && (
        <div className="panel">
          <div className="panel-title">Batch result &middot; <span className={`badge ${result.batch.status === 'COMPLETED' ? 'badge-good' : 'badge-bad'}`}>{result.batch.status}</span></div>
          {result.batch.status === 'COMPLETED' ? (
            <div className="metric-row">
              <div className="metric">records <b>{result.batch.record_count}</b></div>
              <div className="metric">attached to existing <b>{result.batch.attached_to_existing_count}</b></div>
              <div className="metric">new entities <b>{result.batch.new_entity_count}</b></div>
              <div className="metric">ambiguous (sent to review) <b>{result.batch.ambiguous_count}</b></div>
              <div className="metric">candidate pairs <b>{result.batch.candidate_pair_count}</b></div>
              <div className="metric">time <b>{result.batch.processing_seconds}s</b></div>
            </div>
          ) : (
            <div className="evidence-drawer">{result.batch.error}</div>
          )}
        </div>
      )}

      <div className="panel-title" style={{ marginTop: 20 }}>Recent batches</div>
      {batches.length === 0 && <div className="spinner-text">No incoming batches processed yet.</div>}
      {batches.map(b => (
        <div key={b.batch_run_id} className="candidate-row">
          <div className="candidate-main">
            <div className="candidate-col">{b.batch_run_id}</div>
            <div className="candidate-sub">
              model {b.model_version} &middot; {b.record_count ?? '—'} records &middot; attached {b.attached_to_existing_count ?? '—'} &middot; new {b.new_entity_count ?? '—'} &middot; ambiguous {b.ambiguous_count ?? '—'}
            </div>
          </div>
          <span className={`badge ${b.status === 'COMPLETED' ? 'badge-good' : b.status === 'FAILED' ? 'badge-bad' : 'badge-neutral'}`}>{b.status}</span>
        </div>
      ))}
    </div>
  );
}
