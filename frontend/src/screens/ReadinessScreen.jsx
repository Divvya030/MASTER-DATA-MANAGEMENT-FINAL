import React, { useEffect, useRef, useState } from 'react';
import { api } from '../api';

// Final ingestion gate. Polls the readiness endpoint while not ready (the
// backend may flip ER_CONFIGURED -> READY_FOR_DEDUPE asynchronously, e.g.
// after a background quality check), refreshes the parent dataset state the
// moment readiness is reported, and gives an explicit primary action in both
// directions -- the user never has to guess that pipeline nodes are clickable.
export default function ReadinessScreen({ dataset, onReady, onContinue, onSkipToMatch, onBackToConfigure }) {
  const [report, setReport] = useState(null);
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);
  // Keep the latest callback in a ref: the polling effect must not depend on
  // the callback's identity (a new closure each parent render would re-trigger
  // the effect and re-invoke onReady in a loop).
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  useEffect(() => {
    let cancelled = false;
    let ready = false;
    async function check() {
      try {
        const r = await api.readiness(dataset.dataset_id);
        if (cancelled) return;
        setReport(r.readiness);
        setConfig(r.dedupe_handoff_configuration);
        setError(null);
        if (r.readiness.is_ready && !ready) {
          ready = true;
          if (onReadyRef.current) onReadyRef.current();
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    }
    check();
    const timer = setInterval(() => {
      // Stop polling once ready: readiness is monotonic for a dataset.
      if (!cancelled && !ready) check();
    }, 4000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [dataset.dataset_id]);

  // Guard against a stale snapshot from a previously selected dataset:
  // only show a report for THIS dataset (readiness is server truth).
  const fresh = report && report.dataset_id === dataset.dataset_id;
  const readyNow = fresh ? report.is_ready : false;

  return (
    <div>
      <div className="page-header">
        <h1>Final pre-dedupe review</h1>
        <p>A transparent, explainable readiness assessment — not a fabricated score. Every check and warning is backed by evidence collected in earlier stages.</p>
      </div>

      {error && <div className="status-banner error">{error}</div>}
      {!report && !error && <div className="spinner-text">Loading readiness assessment…</div>}

      {fresh && report && (
        <>
          <div className={`readiness-hero ${readyNow ? 'ready' : ''}`}>
            <h2>{readyNow ? '✓ Ready for Dedupe' : 'Not ready yet'}</h2>
            <div className="rh-sub">{report.dataset_id} · {report.domain} · {report.row_count.toLocaleString()} rows</div>
            <div className="check-list">
              {report.checks.map((c, i) => (
                <div className={`check-item ${c.passed ? 'pass' : 'fail'}`} key={i}>
                  <span className="check-mark">{c.passed ? '✓' : '✗'}</span>
                  <span>{c.name}{c.detail ? ` — ${c.detail}` : ''}</span>
                </div>
              ))}
            </div>
            {report.warnings.length > 0 && (
              <div className="warning-list">
                {report.warnings.map((w, i) => <div className="warning-item" key={i}>⚠ {w}</div>)}
              </div>
            )}
          </div>

          <div className="panel" style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            {readyNow ? (
              <>
                <span style={{ fontWeight: 700 }}>✓ Dataset is ready for entity resolution.</span>
                <button className="btn btn-primary" data-testid="readiness-continue" onClick={onContinue}>
                  Continue to Training &amp; Labeling (v2) →
                </button>
                {onSkipToMatch && (
                  <button className="btn" data-testid="readiness-skip-to-match" onClick={onSkipToMatch}>
                    Match now with the ACTIVE V1 model →
                  </button>
                )}
              </>
            ) : (
              <>
                <span style={{ fontWeight: 700 }}>Not ready yet — fix the failing checks above.</span>
                {onBackToConfigure && (
                  <button className="btn" data-testid="readiness-back" onClick={onBackToConfigure}>
                    ← Return to ER Configuration
                  </button>
                )}
              </>
            )}
          </div>

          <div className="panel">
            <div className="panel-title">Dataset readiness snapshot</div>
            <div className="metric-row">
              <div className="metric">fields <b>{report.field_count}</b></div>
              <div className="metric">mapped <b>{report.mapped_field_count}</b></div>
              <div className="metric">unmapped <b>{report.unmapped_field_count}</b></div>
              <div className="metric">empty <b>{report.empty_field_count}</b></div>
            </div>
            {report.unmapped_canonical_fields.length > 0 && (
              <div className="empty-cols-box" style={{ marginTop: 12 }}>
                {report.unmapped_canonical_fields.length} canonical field(s) have no source column in this dataset at all — this reflects data the source system never collected, not a mapping limitation:
                <div>
                  {report.unmapped_canonical_fields.map(f => <span key={f} className="chip">{f}</span>)}
                </div>
              </div>
            )}
          </div>

          <div className="panel">
            <div className="panel-title">Dedupe hand-off configuration</div>
            <div className="two-col">
              <div>
                <div className="rail-section-label" style={{ color: 'var(--slate)' }}>IDENTIFIER</div>
                {config.identifier_fields.length ? config.identifier_fields.map(f => <div key={f} className="field-name">{f}</div>) : <div className="metric">none</div>}
                <div className="rail-section-label" style={{ color: 'var(--slate)', marginTop: 12 }}>MATCH</div>
                {config.match_fields.map(f => <div key={f} className="field-name">{f}</div>)}
              </div>
              <div>
                <div className="rail-section-label" style={{ color: 'var(--slate)' }}>IGNORED</div>
                {config.ignored_fields.length ? config.ignored_fields.map(f => <div key={f} className="field-name">{f}</div>) : <div className="metric">none</div>}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
