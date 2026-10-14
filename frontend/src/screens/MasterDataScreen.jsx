import React, { useEffect, useState, useCallback, useRef } from 'react';
import { api } from '../api';
import { useDomainSchema, orderedEntries, labelFor, isBlank } from '../fieldOrder';
import { SummaryChips } from '../components/RecordFields';

const PAGE = 50;

const REASON_BADGE = {
  NEEDS_REVIEW: { cls: 'badge-warn', text: 'Conflicts to resolve' },
  PENDING_APPROVAL: { cls: 'badge-neutral', text: 'Awaiting approval' },
  RE_APPROVAL: { cls: 'badge-warn', text: 'Changed since approval' },
};

// FINAL MASTER DATA: one current Golden Record per entity. Single-record
// ("distinct") entities -- unmatched records and steward-confirmed DISTINCT
// pairs -- are first-class members and appear here automatically; merged
// entities appear once a steward approves them. Everything that is NOT final yet
// is listed under "Awaiting approval", so no entity is ever invisible.
export default function MasterDataScreen({ domain, onOpenEntity }) {
  const schema = useDomainSchema(domain);
  const [scope, setScope] = useState('final');      // final | awaiting
  const [kind, setKind] = useState('all');          // all | merged | distinct
  const [search, setSearch] = useState('');
  const [query, setQuery] = useState('');           // debounced search actually sent
  const [records, setRecords] = useState([]);
  const [summary, setSummary] = useState(null);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState({});
  const [error, setError] = useState(null);
  const seq = useRef(0);   // ignore responses that belong to an older filter

  useEffect(() => {
    const t = setTimeout(() => setQuery(search), 300);
    return () => clearTimeout(t);
  }, [search]);

  const load = useCallback(async (offset, replace) => {
    const my = ++seq.current;
    setLoading(true); setError(null);
    try {
      const r = await api.masterData(domain, offset, PAGE, { scope, kind, q: query });
      if (my !== seq.current) return;
      setSummary(r);
      setHasMore(!!r.has_more);
      setRecords(prev => replace ? r.records : [...prev, ...r.records]);
    } catch (e) { if (my === seq.current) setError(e.message); } finally { if (my === seq.current) setLoading(false); }
  }, [domain, scope, kind, query]);

  useEffect(() => { load(0, true); }, [load]);

  const total = summary ? (summary.matching ?? summary.count) : 0;
  const awaiting = summary?.awaiting_count ?? summary?.awaiting_review_count ?? 0;

  return (
    <div>
      <div className="page-header">
        <h1>Final master data</h1>
        <p>The enterprise representation of each entity. Distinct (single-record) entities are included
          automatically because there is nothing to reconcile; merged entities appear once a steward approves them,
          and any entity with unresolved conflicts stays in Steward Review. Records that are not final yet are listed
          under <b>Awaiting approval</b>. Field lineage (source record, source system, rule) stays available on the
          Entities screen.</p>
      </div>
      {error && <div className="status-banner error">{error}</div>}

      {summary && (
        <div className="panel" style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
          <div className="metric">final master records <b>{summary.count.toLocaleString()}</b></div>
          <div className="metric">distinct (single-record) <b>{(summary.distinct_record_count ?? 0).toLocaleString()}</b></div>
          <div className="metric">merged &amp; approved <b>{(summary.merged_record_count ?? 0).toLocaleString()}</b></div>
          <div className="metric">awaiting review / approval <b>{awaiting.toLocaleString()}</b></div>
        </div>
      )}

      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', margin: '4px 0 14px' }}>
        <div className="btn-row" role="tablist" aria-label="Master data view">
          <button role="tab" aria-selected={scope === 'final'} className={`btn btn-sm${scope === 'final' ? ' btn-primary' : ''}`}
                  onClick={() => setScope('final')}>Final records</button>
          <button role="tab" aria-selected={scope === 'awaiting'} className={`btn btn-sm${scope === 'awaiting' ? ' btn-primary' : ''}`}
                  onClick={() => setScope('awaiting')}>Awaiting approval{summary ? ` (${awaiting.toLocaleString()})` : ''}</button>
        </div>
        <select aria-label="Record type" value={kind} onChange={e => setKind(e.target.value)} style={{ padding: 6, borderRadius: 4, fontSize: 12.5 }}>
          <option value="all">All records</option>
          <option value="merged">Merged (2+ source records)</option>
          <option value="distinct">Single-record</option>
        </select>
        <input className="field-input" style={{ maxWidth: 280 }} placeholder="Search name, id, city, email…"
               aria-label="Search master data" value={search} onChange={e => setSearch(e.target.value)} />
        <span className="metric">{loading && records.length === 0 ? 'Loading…' : <><b>{total.toLocaleString()}</b> matching</>}</span>
      </div>

      {loading && records.length === 0 && !error && <div className="spinner-text">Loading…</div>}

      {summary && summary.count === 0 && scope === 'final' && !loading && !query && kind === 'all' && (
        <div className="spinner-text">
          No final master records yet. Run matching, then resolve conflicts and approve merged records on the Steward Review / Entities screens.
        </div>
      )}
      {summary && !loading && records.length === 0 && (query || kind !== 'all' || scope === 'awaiting') && (
        <div className="spinner-text">
          {scope === 'awaiting' ? 'Nothing is waiting for approval.' : 'No master record matches this filter.'}
        </div>
      )}

      {records.map(r => {
        const reason = REASON_BADGE[r.awaiting_reason];
        const rows = orderedEntries(r.values, schema).filter(([, v]) => !isBlank(v));
        return (
          <div className="field-card" key={r.entity_id}>
            <div className="field-card-head">
              <span className="field-name">{r.golden_record_id}</span>
              <div>
                {scope === 'final'
                  ? <span className="badge badge-good">{r.approval_basis === 'STEWARD_APPROVED' ? 'Approved' : 'Distinct record'}</span>
                  : reason && <span className={`badge ${reason.cls}`}>{reason.text}</span>}
                <span className="badge badge-neutral" style={{ marginLeft: 6 }}>v{r.version} · {r.member_count} record{r.member_count === 1 ? '' : 's'}</span>
              </div>
            </div>
            {r.display_name && <div style={{ fontWeight: 700, marginTop: 6 }}>{r.display_name}</div>}
            {r.summary?.length > 0
              ? <SummaryChips items={r.summary.filter(it => it.value !== r.display_name)} />
              : (
                <div className="metric-row" style={{ marginTop: 8 }}>
                  {rows.slice(0, 6).map(([k, v]) => (
                    <div className="metric" key={k}>{labelFor(k, schema)} <b>{String(v).slice(0, 40)}</b></div>
                  ))}
                </div>
              )}
            {scope === 'awaiting' && r.awaiting_detail && (
              <div className="evidence-drawer" style={{ marginTop: 8, fontSize: 12 }}>{r.awaiting_detail}</div>
            )}
            <div className="btn-row" style={{ marginTop: 8 }}>
              <button className="btn btn-sm"
                      onClick={() => setExpanded(prev => ({ ...prev, [r.entity_id]: !prev[r.entity_id] }))}>
                {expanded[r.entity_id] ? 'Hide all fields' : 'Show all fields'}
              </button>
              {onOpenEntity && (
                <button className="btn btn-sm" onClick={() => onOpenEntity(r.entity_id)}>
                  {scope === 'awaiting' ? 'Review & approve →' : 'Open entity →'}
                </button>
              )}
            </div>
            {expanded[r.entity_id] && (
              <div className="evidence-drawer" style={{ marginTop: 8 }}>
                {rows.map(([k, v]) => (
                  <div key={k} className="kv-row" style={{ fontSize: 12.5, marginBottom: 4 }}>
                    <span className="k">{labelFor(k, schema)}:</span> <span style={{ minWidth: 0, wordBreak: 'break-word' }}>{String(v)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {records.length > 0 && (
        <div className="btn-row" style={{ gap: 12, margin: '8px 0' }}>
          <span className="metric">showing <b>{records.length.toLocaleString()}</b> of <b>{total.toLocaleString()}</b></span>
          {hasMore && (
            <button className="btn btn-sm" disabled={loading} onClick={() => load(records.length, false)}>
              {loading ? 'Loading…' : `Load ${PAGE} more`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
