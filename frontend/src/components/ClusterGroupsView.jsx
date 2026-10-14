import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../api';

// Bounded, paginated view of the multi-record clusters of ONE cluster run, with the actual source
// records of each cluster. Cluster membership is computed by the backend (Dedupe clustering); this
// only displays it. Each request returns at most `pageSize` groups and only those groups' records.
//
// Layout: every cluster is a card. Opening it shows a field-by-field comparison -- one COLUMN per
// source record, one ROW per field -- so a steward can scan across and immediately see which fields
// agree (dimmed) and which disagree (highlighted). The record's source system and last-updated time
// (the inputs to survivorship) sit in the column header instead of being buried as two more fields.

const scoreClass = (v) => (v >= 0.9 ? 'good' : v >= 0.7 ? 'warn' : 'bad');
const prettyField = (c) => String(c).replace(/_/g, ' ');
const isBlank = (v) => v == null || String(v).trim() === '';
const toMs = (v) => {
  if (isBlank(v)) return null;
  const t = Date.parse(String(v).trim().replace(' ', 'T'));
  return Number.isNaN(t) ? null : t;
};

export default function ClusterGroupsView({ datasetId, clusterRunId, pageSize = 25 }) {
  const [data, setData] = useState(null);          // { total_groups, columns, groups, ... }
  const [open, setOpen] = useState({});
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState('standardized');   // what the model compared (default) | raw source values
  const [onlyDiff, setOnlyDiff] = useState(false);
  const [query, setQuery] = useState('');
  const [error, setError] = useState(null);

  const loadPage = useCallback(async (offset) => {
    setLoading(true); setError(null);
    try {
      const g = await api.duplicateGroups(datasetId, clusterRunId, offset, pageSize);
      setData(prev => (offset === 0 || !prev) ? g : { ...g, groups: [...prev.groups, ...g.groups] });
      if (offset === 0 && g.groups.length > 0) setOpen({ [g.groups[0].group_id]: true });
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  }, [datasetId, clusterRunId, pageSize]);

  useEffect(() => { setData(null); setOpen({}); loadPage(0); }, [loadPage]);

  const model = useMemo(() => {
    if (!data) return null;
    const useStd = view === 'standardized' && !!data.standardized_columns;
    const metaFields = data.metadata_fields || [];
    const mapping = data.mapping || {};
    const sysField = data.source_system_field || 'source_system';
    const recField = data.recency_field || 'last_updated';
    // columns shown as comparison rows: everything except record metadata (shown in the header)
    const hidden = new Set(useStd ? metaFields : metaFields.map(f => mapping[f]).filter(Boolean));
    const cols = (useStd ? data.standardized_columns : data.columns).filter(c => !hidden.has(c));
    const cell = (r, c) => (useStd ? r._std?.[c] : r[c]);
    const meta = (r) => ({ system: r._std?.[sysField] || null, updated: r._std?.[recField] || null });

    // stable colour per source system, in order of first appearance
    const systems = [];
    for (const g of data.groups) for (const r of g.records) {
      const s = meta(r).system;
      if (s && !systems.includes(s)) systems.push(s);
    }
    const chipClass = (s) => `s${Math.max(0, systems.indexOf(s)) % 5}`;

    const analysed = data.groups.map(g => {
      const rows = cols.map(c => {
        const vals = g.records.map(r => cell(r, c));
        const distinct = new Set(vals.filter(v => !isBlank(v)).map(v => String(v).trim()));
        return { c, vals, differs: distinct.size > 1, empty: distinct.size === 0 };
      });
      const stamps = g.records.map(r => toMs(meta(r).updated));
      const valid = stamps.filter(t => t != null);
      const newest = valid.length > 1 && new Set(valid).size > 1 ? Math.max(...valid) : null;
      const nameRow = rows.find(x => /name/i.test(x.c) && x.vals.some(v => !isBlank(v)))
        || rows.find(x => x.vals.some(v => !isBlank(v)));
      const label = nameRow ? String(nameRow.vals.find(v => !isBlank(v))) : '';
      const sys = [...new Set(g.records.map(r => meta(r).system).filter(Boolean))];
      const text = g.records.flatMap(r => cols.map(c => cell(r, c))).filter(v => !isBlank(v)).join(' ').toLowerCase();
      return { g, rows, stamps, newest, label, sys, diffCount: rows.filter(x => x.differs).length, text };
    });
    return { cols, cell, meta, chipClass, analysed, hasMeta: systems.length > 0 || analysed.some(a => a.stamps.some(t => t != null)) };
  }, [data, view]);

  if (error) return <div className="status-banner error">Could not load cluster records: {error}</div>;
  if (!data || !model) return <div className="spinner-text">Loading clustered records…</div>;

  const shown = data.groups.length;
  const q = query.trim().toLowerCase();
  const visible = q ? model.analysed.filter(a => a.text.includes(q) || a.g.group_id.toLowerCase().includes(q)) : model.analysed;
  const allOpen = visible.length > 0 && visible.every(a => open[a.g.group_id]);
  const toggle = (id) => setOpen(o => ({ ...o, [id]: !o[id] }));

  return (
    <div data-testid="cluster-groups">
      <div className="cluster-toolbar">
        <label>Show values
          <select value={view} onChange={e => setView(e.target.value)} data-testid="cluster-view-select">
            <option value="standardized">Standardized (what the model compared)</option>
            <option value="raw">Raw source values</option>
          </select>
        </label>
        <label>
          <input type="checkbox" checked={onlyDiff} onChange={e => setOnlyDiff(e.target.checked)} />
          Only fields that differ
        </label>
        <input type="search" placeholder="Filter loaded clusters…" value={query}
               onChange={e => setQuery(e.target.value)} aria-label="Filter loaded clusters" />
        <div className="btn-row" style={{ marginLeft: 'auto' }}>
          <button className="btn btn-sm" disabled={visible.length === 0}
                  onClick={() => setOpen(o => {
                    const n = { ...o };
                    for (const a of visible) n[a.g.group_id] = !allOpen;
                    return n;
                  })}>
            {allOpen ? 'Collapse all' : 'Expand all'}
          </button>
        </div>
      </div>

      <p className="cluster-summary">
        {data.total_groups === 0
          ? 'No multi-record clusters at this threshold — every record is its own entity.'
          : `Showing 1–${shown} of ${data.total_groups.toLocaleString()} multi-record clusters (largest and least-certain first). Single-record clusters are counted above but not listed.`}
        {q && visible.length !== shown ? ` ${visible.length} of the loaded clusters match “${query.trim()}”.` : ''}
      </p>

      {visible.map(({ g, rows, stamps, newest, label, sys, diffCount }) => {
        const isOpen = !!open[g.group_id];
        const score = g.match_score ?? g.confidence;
        const cls = scoreClass(score);
        const shownRows = onlyDiff ? rows.filter(x => x.differs) : rows;
        return (
          <div key={g.group_id} className={`cluster-card score-${cls}${isOpen ? ' open' : ''}`}>
            <div className="cluster-head" role="button" tabIndex={0} aria-expanded={isOpen}
                 data-testid={`cluster-header-${g.group_id}`}
                 onClick={() => toggle(g.group_id)}
                 onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(g.group_id); } }}>
              <span className="cluster-chevron" aria-hidden="true">▼</span>
              <b className="cluster-id">Cluster {g.group_id}</b>
              {label ? <span className="cluster-label" title={label}>{label}</span> : <span className="cluster-label" />}
              <span className="cluster-meta">
                {sys.map(s => <span key={s} className={`src-chip ${model.chipClass(s)}`}>{s}</span>)}
                <span className="count-pill">{g.size} records</span>
                <span className={`count-pill ${diffCount ? 'diff' : 'same'}`}
                      title="Fields where records hold different non-empty values">
                  {diffCount ? `${diffCount} field${diffCount === 1 ? '' : 's'} differ` : 'all fields agree'}
                </span>
                <span className={`score-pill ${cls}`}
                      title="Lowest pair score inside the cluster (weakest link). A model score, not a probability.">
                  <span className="bar"><i style={{ width: `${Math.max(4, Math.min(100, score * 100))}%` }} /></span>
                  match score {score.toFixed(2)}
                </span>
              </span>
            </div>

            {isOpen && (
              <div className="cluster-body">
                <div className="cluster-body-note">
                  <span className="legend"><span className="swatch" /> values that differ between records</span>
                  {model.hasMeta && <span>Column headers show each record’s source system and last-updated time — the inputs survivorship uses.</span>}
                </div>
                <div className="cmp-wrap">
                  <table className="cmp-table">
                    <thead>
                      <tr>
                        <th className="cmp-field" scope="col">Field</th>
                        {g.records.map((r, i) => {
                          const m = model.meta(r);
                          const isNewest = newest != null && stamps[i] === newest;
                          return (
                            <th key={r.row_number} scope="col">
                              <div className="cmp-rec-head">
                                <span className="row-no">Row {r.row_number}</span>
                                {m.system && <span className={`src-chip ${model.chipClass(m.system)}`}>{m.system}</span>}
                                {m.updated && <span className="time-chip">{m.updated}</span>}
                                {isNewest && <span className="newest">● most recent</span>}
                              </div>
                            </th>
                          );
                        })}
                      </tr>
                    </thead>
                    <tbody>
                      {shownRows.length === 0 && (
                        <tr><td className="cell-same" colSpan={g.records.length + 1}>All fields agree across these records.</td></tr>
                      )}
                      {shownRows.map(x => (
                        <tr key={x.c} className={x.differs ? '' : 'cmp-row-same'}>
                          <th className="cmp-field" scope="row">{prettyField(x.c)}</th>
                          {x.vals.map((v, i) => (
                            <td key={g.records[i].row_number}
                                className={isBlank(v) ? 'cell-missing' : x.differs ? 'cell-diff' : 'cell-same'}>
                              {isBlank(v) ? '—' : v}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        );
      })}

      {q && visible.length === 0 && <div className="cluster-summary">No loaded cluster matches “{query.trim()}”.</div>}

      {shown < data.total_groups && (
        <button className="btn btn-sm" style={{ marginTop: 10 }} disabled={loading} onClick={() => loadPage(shown)}>
          {loading ? 'Loading…' : `Show more (${(data.total_groups - shown).toLocaleString()} left)`}
        </button>
      )}
    </div>
  );
}
