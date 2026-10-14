import React, { useEffect, useState } from 'react';
import { api } from '../api';

const REC_BADGE = {
  GOOD_MATCH_FIELD: 'badge-good',
  STRONG_IDENTIFIER_CANDIDATE: 'badge-good',
  SUPPORTING_MATCH_FIELD: 'badge-warn',
  LOW_VALUE_FOR_ER: 'badge-bad',
  INSUFFICIENT_DATA: 'badge-bad',
};

const REC_PLAIN = {
  GOOD_MATCH_FIELD: 'Strong match field',
  STRONG_IDENTIFIER_CANDIDATE: 'Looks like an identifier',
  SUPPORTING_MATCH_FIELD: 'Supporting match field',
  LOW_VALUE_FOR_ER: 'Low value for matching',
  INSUFFICIENT_DATA: 'Not enough data',
};

export default function ProfilingScreen({ dataset, onProfiled }) {
  const [profiles, setProfiles] = useState(null);
  const [expanded, setExpanded] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getProfile(dataset.dataset_id, 'RAW').then(r => setProfiles(r.profiles)).catch(() => setProfiles(null));
  }, [dataset.dataset_id]);

  async function runProfiling() {
    setLoading(true);
    setError(null);
    try {
      const r = await api.runProfile(dataset.dataset_id);
      setProfiles(r.profiles);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-header">
        <h1>Data quality profile</h1>
        <p>What the data says about each field — in plain language, with the full statistics underneath.</p>
      </div>

      {error && <div className="status-banner error">{error}</div>}

      {!profiles && (
        <div className="panel">
          <button className="btn btn-primary" disabled={loading} onClick={runProfiling}>
            {loading ? 'Profiling…' : 'Run profiling'}
          </button>
        </div>
      )}

      {profiles && Object.values(profiles).map(p => {
        const dq = p.data_quality;
        const completeness = (100 - dq.completeness.null_percentage).toFixed(1);
        const key = p.canonical_field;
        const missingVariants = Object.keys(dq.completeness.missing_token_breakdown || {})
          .filter(k => k && k.trim());
        return (
          <div className="field-card" key={key}>
            <div className="field-card-head">
              <span className="field-name">{p.canonical_field}</span>
              <span className={`badge ${REC_BADGE[p.er_recommendation] || 'badge-neutral'}`}>
                {REC_PLAIN[p.er_recommendation] || p.er_recommendation}
              </span>
            </div>
            <p style={{ margin: '8px 0', fontSize: 13.5, color: 'var(--fg)' }}>
              {p.steward_explanation || (p.reasons && p.reasons[0])}
            </p>
            <div className="metric-row">
              <div className="metric">filled <b>{completeness}%</b></div>
              <div className="metric">distinct <b>{dq.cardinality.uniqueness_ratio}</b></div>
              <div className="metric">type <b>{dq.type_inference.inferred_type}</b> ({dq.type_inference.type_confidence}%)</div>
              <div className="metric">readiness <b>{p.er_readiness}</b></div>
            </div>
            <div>
              {p.cleansing_signals.map(s => <span key={s} className="signal-chip">{s.replaceAll('_', ' ').toLowerCase()}</span>)}
            </div>
            <button className="btn btn-sm" style={{ marginTop: 8 }}
              onClick={() => setExpanded(prev => ({ ...prev, [key]: !prev[key] }))}>
              {expanded[key] ? 'Hide evidence' : 'Show evidence'}
            </button>
            {expanded[key] && (
              <div className="evidence-drawer">
                <div><b>Why:</b></div>
                <ul style={{ margin: '4px 0 8px 18px', padding: 0 }}>
                  {p.reasons.map((r, i) => <li key={i}>{r}</li>)}
                </ul>
                <div><b>Top values:</b> {dq.cardinality.top_values.map(v => `${v.value} (${v.count})`).join(', ')}</div>
                {(missingVariants.length > 0 || dq.completeness.null_count > 0) && (
                  <div style={{ marginTop: 4 }}>
                    <b>Missing values:</b> {dq.completeness.null_count} empty rows
                    {missingVariants.length > 0 && (
                      <> — placeholder tokens found: {missingVariants.map(v => `"${v}"`).join(', ')
                        } ({missingVariants.map(v => dq.completeness.missing_token_breakdown[v]).join(', ')})</>
                    )}
                  </div>
                )}
                {dq.identifier_quality && (
                  <div style={{ marginTop: 4 }}>
                    <b>Identifier quality:</b> format consistency {dq.identifier_quality.format_consistency},
                    {' '}scientific notation count {dq.identifier_quality.scientific_notation_count}
                  </div>
                )}
                {dq.date_ambiguity && (
                  <div style={{ marginTop: 4 }}>
                    <b>Date ambiguity:</b> ambiguous={String(dq.date_ambiguity.is_ambiguous)},
                    {' '}dd/mm evidence {dq.date_ambiguity.dd_mm_evidence_count}, mm/dd evidence {dq.date_ambiguity.mm_dd_evidence_count}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}

      {profiles && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
          <button className="btn btn-primary" onClick={onProfiled}>Continue to standardization →</button>
        </div>
      )}
    </div>
  );
}
