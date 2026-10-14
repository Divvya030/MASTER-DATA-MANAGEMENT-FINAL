import React, { useEffect, useState } from 'react';
import { api } from '../api';

const ROLES = ['MATCH', 'IDENTIFIER', 'IGNORE'];

const ROLE_HELP = {
  MATCH: 'Used to compare records and decide if they are the same entity.',
  IDENTIFIER: 'A business key (e.g. tax id, DUNS) shown as evidence, not scored by the model.',
  IGNORE: 'Left out of matching entirely.',
};

const REC_PLAIN = {
  GOOD_MATCH_FIELD: 'strong match field',
  STRONG_IDENTIFIER_CANDIDATE: 'looks like an identifier',
  SUPPORTING_MATCH_FIELD: 'supporting match field',
  LOW_VALUE_FOR_ER: 'low value for matching',
  INSUFFICIENT_DATA: 'not enough data',
};

export default function ERConfigScreen({ dataset, onConfigured }) {
  const [fields, setFields] = useState(null);
  const [roles, setRoles] = useState({}); // canonical_field -> array of active roles
  const [error, setError] = useState(null);
  const [warnings, setWarnings] = useState([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api.erConfigSuggestions(dataset.dataset_id).then(r => {
      setFields(r.fields);
      const initial = {};
      Object.values(r.fields).forEach(f => { initial[f.canonical_field] = [...f.suggested_roles]; });
      setRoles(initial);
    }).catch(e => setError(e.message));
  }, [dataset.dataset_id]);

  // IGNORE is exclusive with every other role: selecting it clears everything
  // else for that field, and selecting any other role clears IGNORE.
  function toggleRole(field, role) {
    setRoles(prev => {
      const current = prev[field] || [];
      let next;
      if (role === 'IGNORE') {
        next = current.includes('IGNORE') ? [] : ['IGNORE'];
      } else if (current.includes(role)) {
        next = current.filter(r => r !== role);
      } else {
        next = [...current.filter(r => r !== 'IGNORE'), role];
      }
      return { ...prev, [field]: next };
    });
  }

  async function submit() {
    setSaving(true);
    setError(null);
    try {
      const assignments = Object.entries(roles).flatMap(([canonical_field, activeRoles]) => {
        const effective = activeRoles.length > 0 ? activeRoles : ['IGNORE'];
        return effective.map(role => ({ canonical_field, role }));
      });
      const r = await api.submitErConfig(dataset.dataset_id, assignments);
      setWarnings(r.warnings || []);
      onConfigured();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  }

  if (!fields) return error ? <div className="status-banner error">{error}</div> : <div className="spinner-text">Loading…</div>;

  return (
    <div>
      <div className="page-header">
        <h1>Configure ER field roles</h1>
        <p>Profiling recommends a role for each field; you confirm it. IGNORE is the one exclusive role — it clears everything else for that field.</p>
      </div>

      {error && <div className="status-banner error">{error}</div>}

      {Object.values(fields).map(f => {
        const active = roles[f.canonical_field] || [];
        return (
          <div className="field-card" key={f.canonical_field}>
            <div className="field-card-head">
              <span className="field-name">{f.canonical_field}</span>
              <span className="badge badge-neutral">suggested: {(f.suggested_roles || []).join(' + ')}</span>
            </div>
            <p style={{ margin: '8px 0', fontSize: 13, color: 'var(--fg)' }}>
              {f.steward_explanation}
            </p>
            <div className="metric-row">
              <div className="metric">filled <b>{f.completeness_pct}%</b></div>
              <div className="metric">profiling <b>{REC_PLAIN[f.er_recommendation] || f.er_recommendation}</b></div>
            </div>
            <div className="role-toggle-group" style={{ marginTop: 10 }}>
              {ROLES.map(r => (
                <div
                  key={r}
                  title={ROLE_HELP[r]}
                  className={`role-toggle ${active.includes(r) ? 'active' : ''}`}
                  onClick={() => toggleRole(f.canonical_field, r)}
                >
                  {r}
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {warnings.length > 0 && (
        <div className="warning-list" style={{ marginTop: 12 }}>
          {warnings.map((w, i) => (
            <div className="warning-item" key={i}>⚠ {w.canonical_field}: {w.message}</div>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
        <button className="btn btn-primary" disabled={saving} onClick={submit}>
          {saving ? 'Saving…' : 'Confirm ER configuration →'}
        </button>
      </div>
    </div>
  );
}
