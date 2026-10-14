import React, { useEffect, useState, useCallback, memo } from 'react';
import { api } from '../api';
import { useDomainSchema } from '../fieldOrder';
import RecordFields, { SummaryChips } from '../components/RecordFields';

const PAGE = 25;

// Memoised so typing the reviewer name (or resolving one item) does not
// re-render every other card in the queue.
const PendingCard = memo(function PendingCard({ p, busy, onResolve, schema }) {
  return (
    <div className="field-card">
      <div className="field-card-head">
        <span className="field-name">{p.candidate_dataset_id} · record {p.candidate_record_id}</span>
        {p.score != null && <span className="badge badge-warn">score {p.score.toFixed(3)}</span>}
      </div>
      <div className="two-col" style={{ marginTop: 10 }}>
        <div className="evidence-drawer">
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Incoming record</div>
          <RecordFields values={p.candidate_values} schema={schema} showMissing />
        </div>
        <div className="evidence-drawer">
          <div style={{ fontWeight: 700, marginBottom: 6 }}>
            {p.candidate_entity ? `Candidate entity ${p.candidate_entity.entity_id}` : `Record ${p.candidate_other_record_id || ''}`}
          </div>
          <RecordFields values={p.candidate_entity?.golden_values || p.candidate_other_values} schema={schema} showMissing />
          {!p.candidate_entity && !p.candidate_other_values && <i style={{ fontSize: 12 }}>ambiguous against another new record in the same batch</i>}
        </div>
      </div>
      <div className="metric-row"><div className="metric">threshold <b>{p.match_threshold}</b></div><div className="metric">review band from <b>{p.review_threshold}</b></div><div className="metric">model <b>{p.model_version}</b></div></div>
      <div className="btn-row" style={{ marginTop: 10 }}>
        <button className="btn btn-sm" disabled={busy} style={{ borderColor: 'var(--good)', color: 'var(--good)' }}
                onClick={() => onResolve(p.pending_review_id, 'CONFIRM_MATCH')}>Confirm match</button>
        <button className="btn btn-sm" disabled={busy} style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}
                onClick={() => onResolve(p.pending_review_id, 'REJECT_MATCH')}>Reject (distinct)</button>
        <button className="btn btn-sm" disabled={busy} onClick={() => onResolve(p.pending_review_id, 'UNSURE')}>Unsure</button>
      </div>
    </div>
  );
});

const NeedsReviewCard = memo(function NeedsReviewCard({ e, onOpenEntity }) {
  return (
    <div className="field-card">
      <div className="field-card-head">
        <span className="field-name">{e.display_name ? `${e.display_name} · ${e.entity_id}` : e.entity_id}</span>
        <span className="badge badge-warn">needs review</span>
      </div>
      <SummaryChips items={(e.summary || []).filter(it => it.value !== e.display_name)} />
      {(e.needs_review_fields || []).map((n, i) => (
        <div key={i} className="warning-item" style={{ marginTop: 6 }}>{n.message}</div>
      ))}
      <div className="evidence-drawer" style={{ marginTop: 8, fontSize: 11.5 }}>
        Open this entity on the Entities screen to resolve, reject, or edit each conflicting field.
      </div>
      {onOpenEntity && <button className="btn btn-sm" style={{ marginTop: 8 }} onClick={() => onOpenEntity(e.entity_id)}>Open entity →</button>}
    </div>
  );
});

// One page of a server-paged list: the first page shows up as soon as it
// arrives, further pages are fetched on demand.
function usePagedList(fetchPage, deps) {
  const [state, setState] = useState({ items: [], total: 0, hasMore: false, loading: true, error: null });

  const load = useCallback(async (offset, replace) => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const r = await fetchPage(offset);
      setState(s => ({
        items: replace ? r.items : [...s.items, ...r.items],
        total: r.total, hasMore: r.hasMore, loading: false, error: null,
      }));
    } catch (e) {
      setState(s => ({ ...s, loading: false, error: e.message }));
    }
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { load(0, true); }, [load]);
  const more = () => load(state.items.length, false);
  const removeById = (pred) => setState(s => {
    const items = s.items.filter(x => !pred(x));
    return { ...s, items, total: Math.max(0, s.total - (s.items.length - items.length)) };
  });
  return { ...state, more, reload: () => load(0, true), removeById };
}

export default function StewardReviewScreen({ domain, onOpenMasterData, onOpenEntity }) {
  const schema = useDomainSchema(domain);
  const [reviewer, setReviewer] = useState('');
  const [outcome, setOutcome] = useState(null);   // what the last decision did to master data
  const [error, setError] = useState(null);
  const [busyId, setBusyId] = useState(null);

  const pending = usePagedList(async (offset) => {
    const r = await api.pendingReviews(domain, offset, PAGE);
    return { items: r.pending_reviews, total: r.total ?? r.count, hasMore: !!r.has_more };
  }, [domain]);

  const needs = usePagedList(async (offset) => {
    const r = await api.entitiesNeedingReview(domain, offset, PAGE);
    return { items: r.entities, total: r.total ?? r.count, hasMore: !!r.has_more };
  }, [domain]);

  const resolve = useCallback(async (id, decision) => {
    if (!reviewer.trim()) { setError('Enter your name/id as reviewer first.'); return; }
    setBusyId(id); setError(null);
    try {
      const r = await api.resolvePendingReview(id, decision, reviewer.trim());
      if (decision !== 'UNSURE') {
        pending.removeById(p => p.pending_review_id === id);
        setOutcome({ decision, ...(r || {}) });
        // a mirrored item (A vs B and B vs A) is answered together with its twin: refresh the queue
        if (r?.mirror_items_resolved > 0) pending.reload();
      } else {
        setOutcome({ decision });
      }
    } catch (e) { setError(e.message); } finally { setBusyId(null); }
  }, [reviewer]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <div className="page-header">
        <h1>Steward Review</h1>
        <p>Ambiguous matches from continuous incoming data, and Golden Records whose conflicting field values
          the survivorship rules could not safely resolve. Every decision here is logged as an audit event and
          becomes reusable labelled-pair evidence.</p>
      </div>
      {(error || pending.error || needs.error) && <div className="status-banner error">{error || pending.error || needs.error}</div>}

      {outcome && <OutcomeBanner o={outcome} onOpenMasterData={onOpenMasterData} onOpenEntity={onOpenEntity}
                                  onClose={() => setOutcome(null)} />}

      <div className="panel">
        <div className="panel-title">Reviewer identity</div>
        <input placeholder="e.g. priya.s" value={reviewer} onChange={e => setReviewer(e.target.value)} className="field-input" />
      </div>

      <div className="panel-title">Match review — are these two records the same entity? ({pending.total.toLocaleString()})</div>
      {pending.loading && pending.items.length === 0 && <div className="spinner-text">Loading…</div>}
      {!pending.loading && pending.items.length === 0 && !pending.error && <div className="spinner-text">Nothing pending.</div>}
      {pending.items.map(p => (
        <PendingCard key={p.pending_review_id} p={p} busy={busyId === p.pending_review_id} onResolve={resolve} schema={schema} />
      ))}
      {pending.items.length > 0 && (
        <div className="btn-row" style={{ gap: 12, margin: '8px 0' }}>
          <span className="metric">showing <b>{pending.items.length.toLocaleString()}</b> of <b>{pending.total.toLocaleString()}</b></span>
          {pending.hasMore && (
            <button className="btn btn-sm" disabled={pending.loading} onClick={pending.more}>
              {pending.loading ? 'Loading…' : `Load ${PAGE} more`}
            </button>
          )}
        </div>
      )}

      <div className="panel-title" style={{ marginTop: 20 }}>Golden Records needing review ({needs.total.toLocaleString()})</div>
      {needs.loading && needs.items.length === 0 && <div className="spinner-text">Loading…</div>}
      {!needs.loading && needs.items.length === 0 && !needs.error && <div className="spinner-text">None pending.</div>}
      {needs.items.map(e => <NeedsReviewCard key={e.entity_id} e={e} onOpenEntity={onOpenEntity} />)}
      {needs.items.length > 0 && (
        <div className="btn-row" style={{ gap: 12, margin: '8px 0' }}>
          <span className="metric">showing <b>{needs.items.length.toLocaleString()}</b> of <b>{needs.total.toLocaleString()}</b></span>
          {needs.hasMore && (
            <button className="btn btn-sm" disabled={needs.loading} onClick={needs.more}>
              {needs.loading ? 'Loading…' : `Load ${PAGE} more`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}


// Tells the steward exactly what their decision did to master data, instead of leaving them to hunt for it.
function OutcomeBanner({ o, onOpenMasterData, onOpenEntity, onClose }) {
  let text;
  if (o.decision === 'UNSURE') {
    text = 'Recorded as UNSURE. The pair stays in the queue and is not used for training.';
  } else if (o.decision === 'REJECT_MATCH') {
    const mine = o.will_appear_in_final_master_data;
    const other = o.counterpart_will_appear_in_final_master_data;
    text = 'Marked DISTINCT - the two records are kept as separate entities.';
    if (mine === true) text += ' The incoming record is in Final Master Data.';
    if (other === true) text += ' The other record is in Final Master Data too.';
    if (other === false) text += ' The other record is part of an entity that is still awaiting approval - find it under Final Master Data → Awaiting approval.';
  } else {
    text = 'Confirmed as a MATCH - the records now share one entity and its Golden Record was rebuilt with the survivorship rules.';
    text += o.will_appear_in_final_master_data
      ? ' It is in Final Master Data.'
      : ' It joins Final Master Data once a steward approves it on the Entities screen.';
  }
  if (o.mirror_items_resolved > 0) text += ' The matching mirrored queue item was closed with it.';
  return (
    <div className="status-banner info" style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
      <span style={{ flex: 1, minWidth: 220 }}>{text}</span>
      {o.resulting_entity_id && onOpenEntity && o.decision === 'CONFIRM_MATCH' &&
        <button className="btn btn-sm" onClick={() => onOpenEntity(o.resulting_entity_id)}>Open entity →</button>}
      {onOpenMasterData && o.decision !== 'UNSURE' && <button className="btn btn-sm" onClick={onOpenMasterData}>Final Master Data →</button>}
      <button className="btn btn-sm" onClick={onClose} aria-label="Dismiss">✕</button>
    </div>
  );
}
