import React, { useEffect, useState, useCallback } from 'react';
import { api } from '../api';

const RULE_INFO = {
  source_priority: {
    text: 'Prefer values from the highest-priority source system (configured below, never hardcoded).',
    needs: 'source',
  },
  recency: {
    text: 'Prefer the value from the most recently updated record. Needs a mapped last-updated field.',
    needs: 'recency',
  },
  validity: { text: 'Prefer a structurally well-formed value (e.g. a valid-looking email or phone).' },
  frequency: { text: 'Prefer the value most member records agree on.' },
  completeness: { text: 'Prefer the longer / more complete value (skipped for identifier-like fields).' },
};

const LIFECYCLE = ['DRAFT', 'STEWARD_CONFIRMED', 'ACTIVE'];

export default function SurvivorshipScreen({ domain }) {
  const [cfg, setCfg] = useState(null);
  const [order, setOrder] = useState([]);
  const [fieldRules, setFieldRules] = useState({}); // {field: [rules]}
  const [newField, setNewField] = useState('');
  const [priorities, setPriorities] = useState([]);  // [{system, priority}]
  const [newSystem, setNewSystem] = useState('');
  const [recencyField, setRecencyField] = useState('');
  const [dayFirst, setDayFirst] = useState(true);
  const [schemaFields, setSchemaFields] = useState([]);
  const [detected, setDetected] = useState(null);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [meta, setMeta] = useState(null);

  const refresh = useCallback(() => {
    api.getSurvivorship(domain).then(r => {
      setCfg(r.config);
      setOrder(r.config.rule_order);
      setFieldRules(r.config.field_rules || {});
      setRecencyField(r.config.recency_field || '');
      setDayFirst(r.config.timestamp_dayfirst !== false);
      const sp = r.config.source_priority || {};
      setPriorities(Object.entries(sp).map(([system, priority]) => ({ system, priority: String(priority) })));
      setMeta({ isDefault: r.is_default, updatedBy: r.updated_by, updatedAt: r.updated_at, version: r.config.version, lifecycle: r.lifecycle_status });
    }).catch(e => setError(e.message));
    // what the uploaded data actually offers survivorship (best-effort; the screen works without it)
    api.detectedSurvivorship?.(domain).then(setDetected).catch(() => setDetected(null));
    api.domainSchema?.(domain).then(r => setSchemaFields(r?.fields || [])).catch(() => setSchemaFields([]));
  }, [domain]);

  useEffect(() => { refresh(); }, [refresh]);

  function move(i, dir) {
    setOrder(prev => {
      const next = [...prev];
      const j = i + dir;
      if (j < 0 || j >= next.length) return prev;
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });
  }

  function moveFieldRule(field, i, dir) {
    setFieldRules(prev => {
      const chain = [...(prev[field] || [])];
      const j = i + dir;
      if (j < 0 || j >= chain.length) return prev;
      [chain[i], chain[j]] = [chain[j], chain[i]];
      return { ...prev, [field]: chain };
    });
  }

  function addFieldRule() {
    const f = newField.trim();
    if (!f) return;
    setFieldRules(prev => (prev[f] ? prev : { ...prev, [f]: [...order] }));
    setNewField('');
  }

  function addSystem(name, priority = '1') {
    const n = name.trim();
    if (!n) return;
    setPriorities(p => (p.some(x => x.system === n) ? p : [...p, { system: n, priority }]));
  }

  async function save() {
    setSaving(true); setError(null);
    try {
      const sourcePriority = {};
      for (const p of priorities) {
        const n = Number(p.priority);
        if (!p.system.trim() || Number.isNaN(n)) {
          setError('Every source system needs a name and a numeric priority.');
          setSaving(false); return;
        }
        sourcePriority[p.system.trim()] = n;
      }
      await api.putSurvivorship(domain, {
        rule_order: order, source_priority: sourcePriority, field_rules: fieldRules,
        recency_field: recencyField || null, timestamp_dayfirst: dayFirst, updated_by: 'steward_ui',
      });
      refresh();
    } catch (e) { setError(e.message); } finally { setSaving(false); }
  }

  async function lifecycleAction(fn) {
    setSaving(true); setError(null);
    try { await fn(domain); refresh(); } catch (e) { setError(e.message); } finally { setSaving(false); }
  }

  if (!cfg) return <div className="spinner-text">Loading…</div>;

  const lifecycle = meta.lifecycle;
  const lifecycleIdx = LIFECYCLE.indexOf(lifecycle);
  const sysField = cfg.source_system_field || 'source_system';
  const sources = detected?.source_systems || [];
  const sourcesMapped = (detected?.datasets || []).some(d => d.source_system_mapped);
  const recencyMapped = (detected?.datasets || []).some(d => d.recency_mapped);
  const parsedPcts = (detected?.datasets || []).map(d => d.timestamps_parsed_pct).filter(v => v != null);
  const worstParsed = parsedPcts.length ? Math.min(...parsedPcts) : null;
  const fieldOptions = schemaFields.length ? schemaFields : (recencyField ? [{ name: recencyField, label: recencyField }] : []);
  const configured = new Set(priorities.map(p => p.system.trim()));
  const ruleReady = (needs) => needs === 'source' ? sourcesMapped : needs === 'recency' ? (!!recencyField && recencyMapped) : true;

  return (
    <div>
      <div className="page-header">
        <h1>Survivorship configuration</h1>
        <p>Field-by-field rules that pick each Golden Record value. Source priority and recency read the
          <b> {sysField}</b> and <b>{recencyField || 'last-updated'}</b> fields of your mapped data — map those columns on the
          Map step and the rules use them automatically. Only an ACTIVE configuration generates production Golden Records.</p>
      </div>
      {error && <div className="status-banner error">{error}</div>}

      <div className="surv-bar">
        <div className="status">
          <div className="lifecycle" title="DRAFT → STEWARD CONFIRMED → ACTIVE">
            {LIFECYCLE.map((l, i) => (
              <React.Fragment key={l}>
                {i > 0 && <i>→</i>}
                <span className={i === lifecycleIdx ? 'on' : ''}>{l.replace('_', ' ')}</span>
              </React.Fragment>
            ))}
          </div>
          {!lifecycle && <span className="badge badge-good">DEFAULT rules are ACTIVE</span>}
          <span className="muted" style={{ fontSize: 12 }}>version {meta.version}{meta.isDefault ? ' · default' : ''}</span>
          {meta.updatedAt && (
            <span className="muted" style={{ fontSize: 11.5 }}>
              updated {new Date(meta.updatedAt).toLocaleString()} by {meta.updatedBy}
            </span>
          )}
        </div>
        <div className="actions">
          <button className="btn btn-primary" disabled={saving} onClick={save}>{saving ? 'Saving…' : 'Save draft'}</button>
          <button className="btn" disabled={saving || lifecycle !== 'DRAFT'}
                  onClick={() => lifecycleAction(api.submitSurvivorship)}>Submit for confirmation</button>
          <button className="btn" disabled={saving || lifecycle !== 'STEWARD_CONFIRMED'}
                  onClick={() => lifecycleAction(api.confirmSurvivorship)}>Activate</button>
        </div>
      </div>

      <div className="input-cards">
        <div className={`input-card ${sourcesMapped ? 'ok' : 'missing'}`}>
          <h3>Source system
            <span className={`badge ${sourcesMapped ? 'badge-good' : 'badge-warn'}`}>{sourcesMapped ? 'mapped' : 'not mapped'}</span>
          </h3>
          <p>{sourcesMapped
            ? `Found ${sources.length} source system${sources.length === 1 ? '' : 's'} in your data: ${sources.map(s => s.name).join(', ') || '—'}.`
            : `No dataset in this domain maps a column to “${sysField}”. Source priority will have no effect until one is mapped (columns named source, source_system, system of record… map automatically).`}</p>
        </div>
        <div className={`input-card ${recencyField && recencyMapped ? 'ok' : 'missing'}`}>
          <h3>Last-updated timestamp
            <span className={`badge ${recencyField && recencyMapped ? 'badge-good' : 'badge-warn'}`}>
              {!recencyField ? 'off' : recencyMapped ? 'mapped' : 'not mapped'}
            </span>
          </h3>
          <div className="field-pick">
            <label htmlFor="recency-field" className="muted">Recency field</label>
            <select id="recency-field" value={recencyField} onChange={e => setRecencyField(e.target.value)}>
              <option value="">(none — recency rule off)</option>
              {fieldOptions.map(f => <option key={f.name} value={f.name}>{f.label || f.name}</option>)}
            </select>
          </div>
          <label className="toggle">
            <input type="checkbox" checked={dayFirst} onChange={e => setDayFirst(e.target.checked)} />
            Read 05/01/2024 as day/month (5 January). Untick for month/day.
          </label>
          {worstParsed != null && (
            <p>{worstParsed >= 99
              ? 'All sampled timestamps parse correctly.'
              : `Only ${worstParsed}% of sampled timestamps could be read — unreadable ones are ignored by the recency rule.`}</p>
          )}
          {!recencyMapped && recencyField && <p>No dataset maps a column to “{recencyField}” yet, so recency has no effect.</p>}
        </div>
      </div>

      <div className="surv-grid">
        <div>
          <div className="panel">
            <div className="panel-title">Default rule order <span className="muted" style={{ fontWeight: 500 }}>applies to every field without its own chain</span></div>
            {order.map((r, i) => {
              const needs = RULE_INFO[r]?.needs;
              const ready = ruleReady(needs);
              return (
                <div key={r} className="rule-step">
                  <span className="no">{i + 1}</span>
                  <div className="txt">
                    <b>{r}</b>
                    <span>{RULE_INFO[r]?.text}</span>
                    {needs && (
                      <span className="needs">
                        <span className={`badge ${ready ? 'badge-good' : 'badge-warn'}`}>
                          {ready ? 'data available' : needs === 'source' ? 'needs a mapped source system' : 'needs a mapped last-updated field'}
                        </span>
                      </span>
                    )}
                  </div>
                  <div className="moves">
                    <button className="btn btn-sm" aria-label={`Move ${r} up`} disabled={i === 0} onClick={() => move(i, -1)}>↑</button>
                    <button className="btn btn-sm" aria-label={`Move ${r} down`} disabled={i === order.length - 1} onClick={() => move(i, 1)}>↓</button>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="panel">
            <div className="panel-title">Field-specific rule chains <span className="muted" style={{ fontWeight: 500 }}>optional overrides</span></div>
            {Object.keys(fieldRules).length === 0 && <div className="metric">No field overrides — every field uses the default order above.</div>}
            {Object.entries(fieldRules).map(([field, chain]) => (
              <div key={field} style={{ marginBottom: 14 }}>
                <div className="btn-row" style={{ marginBottom: 6 }}>
                  <span className="field-name">{field}</span>
                  <button className="btn btn-sm" onClick={() => setFieldRules(prev => { const n = { ...prev }; delete n[field]; return n; })}>remove</button>
                </div>
                {chain.map((r, i) => (
                  <div key={r} className="rule-step">
                    <span className="no">{i + 1}</span>
                    <div className="txt"><b>{r}</b></div>
                    <div className="moves">
                      <button className="btn btn-sm" aria-label={`Move ${r} up for ${field}`} disabled={i === 0} onClick={() => moveFieldRule(field, i, -1)}>↑</button>
                      <button className="btn btn-sm" aria-label={`Move ${r} down for ${field}`} disabled={i === chain.length - 1} onClick={() => moveFieldRule(field, i, 1)}>↓</button>
                    </div>
                  </div>
                ))}
              </div>
            ))}
            <div className="btn-row" style={{ marginTop: 8 }}>
              <input className="field-input" style={{ flex: '1 1 200px' }} value={newField} placeholder="Add field override (e.g. city)…"
                     list="surv-fields" onChange={e => setNewField(e.target.value)}
                     onKeyDown={e => { if (e.key === 'Enter') addFieldRule(); }} />
              <datalist id="surv-fields">
                {schemaFields.filter(f => !f.metadata).map(f => <option key={f.name} value={f.name} />)}
              </datalist>
              <button className="btn btn-sm" onClick={addFieldRule}>+ Add field</button>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">Source system priority <span className="muted" style={{ fontWeight: 500 }}>higher number wins</span></div>
          <p className="muted" style={{ fontSize: 12.5, marginTop: -6, lineHeight: 1.5 }}>
            Each record’s source system is looked up here. A system that is not listed gets priority 0, so a new source
            works with no code change.
          </p>

          {sources.length > 0 && (
            <>
              <div className="muted" style={{ fontSize: 11.5, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.8 }}>Found in your data</div>
              <div className="detected-list">
                {sources.map(s => (
                  <button key={s.name} className="detected-btn" disabled={configured.has(s.name)}
                          title={configured.has(s.name) ? 'Already in the list' : 'Add with default priority 1'}
                          onClick={() => addSystem(s.name)}>
                    {configured.has(s.name) ? '✓' : '+'} {s.name} <small>{s.records.toLocaleString()} records</small>
                  </button>
                ))}
              </div>
            </>
          )}

          {priorities.length === 0 && <div className="metric">No source systems configured — every source is treated equally.</div>}
          {priorities.map((p, i) => (
            <div key={i} className="priority-row">
              <input value={p.system} placeholder="e.g. SAP S/4HANA" aria-label="Source system name"
                     onChange={e => setPriorities(prev => prev.map((x, j) => j === i ? { ...x, system: e.target.value } : x))} />
              <input value={p.priority} type="number" step="1" aria-label={`Priority for ${p.system || 'source system'}`}
                     onChange={e => setPriorities(prev => prev.map((x, j) => j === i ? { ...x, priority: e.target.value } : x))} />
              <button className="btn btn-sm" aria-label={`Remove ${p.system}`} onClick={() => setPriorities(prev => prev.filter((_, j) => j !== i))}>✕</button>
            </div>
          ))}
          <div className="priority-row" style={{ marginTop: 10 }}>
            <input value={newSystem} placeholder="Add source system…"
                   onChange={e => setNewSystem(e.target.value)}
                   onKeyDown={e => { if (e.key === 'Enter') { addSystem(newSystem); setNewSystem(''); } }} />
            <button className="btn btn-sm" onClick={() => { addSystem(newSystem); setNewSystem(''); }}>+ Add</button>
          </div>
        </div>
      </div>
    </div>
  );
}
