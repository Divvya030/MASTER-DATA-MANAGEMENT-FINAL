import React, { useEffect, useState } from 'react';
import { api } from '../api';

export default function MappingScreen({ dataset, onMapped }) {
  const [suggestions, setSuggestions] = useState(null);
  const [chosen, setChosen] = useState({}); // canonical_field -> source_column
  const [expanded, setExpanded] = useState({}); // canonical_field -> bool
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.mappingSuggestions(dataset.dataset_id)
      .then(async (r) => {
        setSuggestions(r);
        try {
          const existing = await api.getMapping(dataset.dataset_id);
          setChosen(existing.mapping || {});
        } catch (e) { /* no mapping yet */ }
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [dataset.dataset_id]);

  if (loading) return <div className="spinner-text">Loading mapping suggestions…</div>;
  if (error) return <div className="status-banner error">{error}</div>;
  if (!suggestions) return null;

  function pick(canonicalField, sourceColumn) {
    setChosen(prev => ({ ...prev, [canonicalField]: sourceColumn }));
  }

  function unassign(canonicalField) {
    setChosen(prev => {
      const next = { ...prev };
      delete next[canonicalField];
      return next;
    });
  }

  async function handleConfirm() {
    setSaving(true);
    setError(null);
    try {
      const mappings = Object.entries(chosen).map(([canonical_field, source_column]) => ({ canonical_field, source_column }));
      await api.confirmMapping(dataset.dataset_id, mappings);
      onMapped();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  }

  const mappedCount = Object.keys(chosen).length;

  return (
    <div>
      <div className="page-header">
        <h1>Map source columns to canonical fields</h1>
        <p>Every candidate shows its own completeness, confidence, and sample values — no need to cross-reference the raw file. Ambiguous fields list every candidate; nothing is auto-selected.</p>
      </div>

      {error && <div className="status-banner error">{error}</div>}

      {suggestions.canonical_fields.map(cf => {
        const candidates = suggestions.suggestions[cf] || [];
        const isExpanded = expanded[cf];
        const currentPick = chosen[cf];
        return (
          <div className="panel" key={cf}>
            <div className="panel-title">
              <span>{cf}</span>
              {candidates.length > 1 && (
                <span className="badge badge-warn">{candidates.length} candidates</span>
              )}
              {candidates.length === 0 && (
                <span className="badge badge-neutral">no candidates</span>
              )}
            </div>

            {candidates.length === 0 && (
              <div className="metric">No source column matched this field with sufficient confidence. You can leave it unmapped.</div>
            )}

            {candidates.map(c => (
              <div key={c.source_column} className={`candidate-row ${currentPick === c.source_column ? 'mapped' : ''}`}>
                <div className="candidate-main" style={{ flex: 1 }}>
                  <div className="candidate-col">{c.source_column}
                    <span className="badge badge-neutral" style={{ marginLeft: 8 }}>{c.match_type}</span>
                    <span className="badge badge-neutral" style={{ marginLeft: 4 }}>{(c.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="candidate-sub">
                    completeness {c.completeness}% · {c.distinct_count.toLocaleString()} distinct values
                  </div>
                  <button
                    className="btn btn-sm"
                    style={{ marginTop: 6, alignSelf: 'flex-start' }}
                    onClick={() => setExpanded(prev => ({ ...prev, [cf + c.source_column]: !prev[cf + c.source_column] }))}
                  >
                    {expanded[cf + c.source_column] ? 'Hide evidence' : 'Show evidence'}
                  </button>
                  {expanded[cf + c.source_column] && (
                    <div className="evidence-drawer">
                      <div><b>Why suggested:</b> {c.reason}</div>
                      <div style={{ marginTop: 6 }}><b>Sample values:</b></div>
                      {c.sample_values.map((v, i) => <span className="sample-value" key={i}>{String(v)}</span>)}
                    </div>
                  )}
                </div>
                <button
                  className={`btn btn-sm ${currentPick === c.source_column ? 'btn-primary' : ''}`}
                  onClick={() => currentPick === c.source_column ? unassign(cf) : pick(cf, c.source_column)}
                >
                  {currentPick === c.source_column ? 'Mapped ✓' : 'Map'}
                </button>
              </div>
            ))}
          </div>
        );
      })}

      {suggestions.empty_ignored_columns.length > 0 && (
        <div className="panel">
          <div className="panel-title">Empty / Ignored columns</div>
          <div className="empty-cols-box">
            These source columns exist in the raw file but contain no values, so they were excluded from mapping candidates. They remain visible here and are never deleted from the raw data.
            <div>
              {suggestions.empty_ignored_columns.map(c => <span key={c} className="chip">{c}</span>)}
            </div>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 16 }}>
        <div className="metric">{mappedCount} of {suggestions.canonical_fields.length} canonical fields mapped</div>
        <button className="btn btn-primary" disabled={saving || mappedCount === 0} onClick={handleConfirm}>
          {saving ? 'Saving…' : 'Confirm mapping →'}
        </button>
      </div>
    </div>
  );
}
