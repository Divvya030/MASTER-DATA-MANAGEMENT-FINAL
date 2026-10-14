import React, { useState, useEffect, useRef } from 'react';
import { api, DuplicateFileError } from '../api';

export default function UploadScreen({ onUploaded }) {
  const [domains, setDomains] = useState([]);
  const [domain, setDomain] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [lastResult, setLastResult] = useState(null);
  const [duplicateChoice, setDuplicateChoice] = useState(null); // { file, existingDatasets }
  const inputRef = useRef(null);

  useEffect(() => {
    api.listDomains().then(r => { setDomains(r.domains); setDomain(r.domains[0]); }).catch(() => {});
  }, []);

  async function handleFile(file, onDuplicate = 'ask') {
    if (!file || !domain) return;
    setBusy(true);
    setError(null);
    try {
      const ds = await api.upload(domain, file, onDuplicate);
      setDuplicateChoice(null);
      setLastResult(ds);
      onUploaded(ds);
    } catch (e) {
      if (e instanceof DuplicateFileError) {
        setDuplicateChoice({ file, existingDatasets: e.existingDatasets });
      } else {
        setError(e.message);
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="page-header">
        <h1>Ingest a source file</h1>
        <p>Upload a customer or supplier extract. The raw file is stored immutably — nothing downstream ever overwrites it. Every upload is fingerprinted with SHA-256 so exact-duplicate files are caught by content, not filename.</p>
      </div>

      <div className="domain-select">
        {domains.map(d => (
          <div key={d} className={`domain-chip ${d === domain ? 'active' : ''}`} onClick={() => setDomain(d)}>
            {d}
          </div>
        ))}
      </div>

      {error && <div className="status-banner error">{error}</div>}

      {duplicateChoice ? (
        <div className="panel">
          <div className="panel-title">This file already exists</div>
          <div className="metric" style={{ marginBottom: 12 }}>
            The content of <b>{duplicateChoice.file.name}</b> exactly matches {duplicateChoice.existingDatasets.length === 1 ? 'a dataset already in the system' : 'datasets already in the system'} (same SHA-256 fingerprint, so this is a true content duplicate — not just a filename match).
          </div>
          {duplicateChoice.existingDatasets.map(d => (
            <div className="candidate-row" key={d.dataset_id}>
              <div className="candidate-main">
                <div className="candidate-col">{d.dataset_id}</div>
                <div className="candidate-sub">{d.original_filename} · {d.row_count.toLocaleString()} rows · {d.status}</div>
              </div>
            </div>
          ))}
          <div style={{ display: 'flex', gap: 10, marginTop: 14 }}>
            <button className="btn btn-primary" onClick={() => handleFile(duplicateChoice.file, 'use_existing')}>
              Use existing dataset
            </button>
            <button className="btn" onClick={() => handleFile(duplicateChoice.file, 'allow_duplicate')}>
              Upload as new anyway
            </button>
          </div>
          <div className="empty-cols-box" style={{ marginTop: 12 }}>
            Delete-and-replace isn't available yet — there's no dataset deletion mechanism in this build. Use one of the two options above, or delete manually at the database level if needed.
          </div>
        </div>
      ) : (
        <div
          className={`upload-zone ${dragging ? 'dragging' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
        >
          <h3>{busy ? 'Ingesting…' : 'Drop a .csv or .xlsx file here'}</h3>
          <p>or click to browse — domain: <b>{domain}</b></p>
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>
      )}

      {lastResult && (
        <div className="panel" style={{ marginTop: 20 }}>
          <div className="panel-title">Ingestion result</div>
          <div className="metric-row">
            <div className="metric"><b>{lastResult.dataset_id}</b></div>
            <div className="metric">{lastResult.original_filename}</div>
            <div className="metric">{lastResult.row_count.toLocaleString()} rows</div>
            <div className="metric">{lastResult.column_count} columns</div>
            <div className="metric">{lastResult.domain}</div>
          </div>
          <div className="metric" style={{ marginTop: 8 }}>sha256: <span className="field-name">{lastResult.sha256?.slice(0, 16)}…</span></div>
          {lastResult.header_warnings?.length > 0 && (
            <div className="empty-cols-box">
              Header warnings: {lastResult.header_warnings.join('; ')}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
