import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { api } from '../api';
import { useDomainSchema, orderedEntries, labelFor, isBlank } from '../fieldOrder';
import { SummaryChips } from '../components/RecordFields';

export default function EntitiesScreen({ domain, initialEntityId = null, onOpenMasterData }) {
  const [entities, setEntities] = useState([]);
  const [total, setTotal] = useState(0);
  const [onlyDuplicates, setOnlyDuplicates] = useState(true);
  const [search, setSearch] = useState('');
  const [selectedId, setSelectedId] = useState(initialEntityId);
  const [reviewer, setReviewer] = useState('');
  const [error, setError] = useState(null);

  const refresh = useCallback(() => {
    const opts = onlyDuplicates ? { min_member_count: 2 } : {};
    api.listEntities(domain, opts).then(r => { setEntities(r.entities); setTotal(r.total_matching); })
      .catch(e => setError(e.message));
  }, [domain, onlyDuplicates]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { if (initialEntityId) setSelectedId(initialEntityId); }, [initialEntityId]);

  // search over entity id, name and every Golden Record value
  const visible = useMemo(() => {
    const t = search.trim().toLowerCase();
    if (!t) return entities;
    return entities.filter(e =>
      e.entity_id.toLowerCase().includes(t) ||
      Object.values(e.golden_record?.values || {}).some(v => !isBlank(v) && String(v).toLowerCase().includes(t)));
  }, [entities, search]);

  if (selectedId) {
    return <EntityDetail entityId={selectedId} domain={domain} reviewer={reviewer} setReviewer={setReviewer}
                         onOpenMasterData={onOpenMasterData}
                         onBack={() => { setSelectedId(null); refresh(); }} />;
  }

  return (
    <div>
      <div className="page-header">
        <h1>Entities &amp; Golden Records</h1>
        <p>Mastered entities for <b>{domain}</b>. Matching decided which records belong together; each entity's
          Golden Record is a separate deterministic step choosing the best value per field.</p>
      </div>
      {error && <div className="status-banner error">{error}</div>}
      <div style={{ marginBottom: 14, display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
        <input className="field-input" style={{ maxWidth: 280 }} placeholder="Search name, id, city, email…"
               aria-label="Search entities" value={search} onChange={e => setSearch(e.target.value)} />
        <label style={{ fontSize: 12.5, display: 'flex', gap: 6, alignItems: 'center' }}>
          <input type="checkbox" checked={onlyDuplicates} onChange={e => setOnlyDuplicates(e.target.checked)} />
          Only entities with more than one source record ({onlyDuplicates ? total : entities.length} matching)
        </label>
        {search.trim() && <span className="metric">showing <b>{visible.length}</b> of <b>{entities.length}</b> loaded</span>}
      </div>
      {entities.length === 0 && <div className="spinner-text">No entities found. Run matching, clustering and golden-record generation first.</div>}
      {entities.length > 0 && visible.length === 0 && <div className="spinner-text">No entity matches “{search}”.</div>}
      {visible.map(e => (
        <div key={e.entity_id} className="candidate-row" style={{ cursor: 'pointer' }} onClick={() => setSelectedId(e.entity_id)}>
          <div className="candidate-main" style={{ minWidth: 0 }}>
            <div className="candidate-col" style={{ fontWeight: 700 }}>
              {e.display_name || e.entity_id}
              {e.display_name && <span style={{ marginLeft: 10, fontWeight: 400, fontSize: 11.5, color: 'var(--slate)' }}>{e.entity_id}</span>}
            </div>
            {e.summary?.length > 0
              ? <SummaryChips items={e.summary.filter(it => it.value !== e.display_name)} />
              : <div className="candidate-sub">{e.golden_record ? '' : 'no golden record'}</div>}
            <div className="candidate-sub">{e.member_count} source record{e.member_count === 1 ? '' : 's'}</div>
          </div>
          <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
            {e.golden_record?.needs_review_fields?.length > 0 && <span className="badge badge-warn">needs review</span>}
            {e.golden_record?.steward_status === 'APPROVED' && <span className="badge badge-good">approved</span>}
            <span className="badge badge-neutral">{e.status}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function EntityDetail({ entityId, domain, onBack, reviewer, setReviewer, onOpenMasterData }) {
  const schema = useDomainSchema(domain);
  const [entity, setEntity] = useState(null);
  const [history, setHistory] = useState(null);
  const [error, setError] = useState(null);
  const [localReviewer, setLocalReviewer] = useState(reviewer || '');
  const [notice, setNotice] = useState(null);

  const refresh = useCallback(() => {
    api.getEntity(entityId).then(setEntity).catch(e => setError(e.message));
    api.getGoldenRecordHistory(entityId).then(r => setHistory(r.versions)).catch(() => {});
  }, [entityId]);

  useEffect(() => { refresh(); }, [refresh]);

  async function act(payload) {
    if (!localReviewer.trim()) { setError('Enter your name/id as reviewer first.'); return; }
    setError(null);
    try {
      await api.submitReview(entityId, { reviewer: localReviewer.trim(), ...payload });
      if (setReviewer) setReviewer(localReviewer.trim());
      setNotice(payload.action === 'APPROVE_GOLDEN_RECORD'
        ? 'Golden Record approved - it is now part of Final Master Data.' : null);
      refresh();
    } catch (e) { setError(e.message); }
  }

  if (!entity) return <div className="spinner-text">Loading…</div>;
  const gr = entity.golden_record;

  return (
    <div>
      <button className="btn btn-sm" onClick={onBack} style={{ marginBottom: 16 }}>&larr; Back to entities</button>
      {error && <div className="status-banner error">{error}</div>}
      {notice && (
        <div className="status-banner info" style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span>{notice}</span>
          {onOpenMasterData && <button className="btn btn-sm" onClick={onOpenMasterData}>Open Final Master Data →</button>}
        </div>
      )}
      <div className="page-header">
        <h1>{entity.display_name || entity.entity_id}</h1>
        <p>{entity.display_name ? <>{entity.entity_id} &middot; </> : null}
          {entity.member_count} source record(s) &middot; status {entity.status} &middot; domain {entity.domain}</p>
      </div>

      <div className="panel">
        <div className="panel-title">Reviewer identity</div>
        <input placeholder="e.g. priya.s" value={localReviewer} onChange={e => setLocalReviewer(e.target.value)} className="field-input" />
      </div>

      {gr && (
        <div className="panel">
          <div className="panel-title">
            Golden Record (version {gr.version_number})
            {gr.needs_review_fields?.length > 0 && <span className="badge badge-warn">needs review</span>}
            {gr.steward_status === 'APPROVED' && <span className="badge badge-good">approved</span>}
          </div>
          {orderedEntries(gr.values, schema).map(([field, value]) => (
            <FieldRow key={field} field={field} label={labelFor(field, schema)} value={value} lineage={gr.lineage[field]}
                      conflicts={gr.conflicts[field]} confidence={gr.field_confidence[field]}
                      needsReview={gr.needs_review_fields?.some(n => n.field === field)}
                      onResolve={v => act({ action: 'RESOLVE_FIELD', field, value: v })}
                      onReject={v => act({ action: 'REJECT_FIELD_VALUE', field, rejected_value: v })}
                      onEdit={v => act({ action: 'EDIT_FIELD', field, value: v })} />
          ))}
          <div style={{ marginTop: 10 }}>
            <button className="btn btn-primary btn-sm" disabled={gr.needs_review_fields?.length > 0}
                    onClick={() => act({ action: 'APPROVE_GOLDEN_RECORD' })}>
              Approve Golden Record
            </button>
            {gr.needs_review_fields?.length > 0 && (
              <span style={{ marginLeft: 10, fontSize: 11.5, color: 'var(--slate)' }}>
                Resolve all conflicting fields before approving.
              </span>
            )}
          </div>
        </div>
      )}

      <div className="panel">
        <div className="panel-title">Source members</div>
        {entity.members.map(m => (
          <div key={`${m.dataset_id}-${m.record_id}`} className="candidate-row">
            <div className="candidate-main">
              <div className="candidate-col">{m.dataset_id} · record {m.record_id}</div>
              <div className="candidate-sub">source: {m.source}</div>
            </div>
          </div>
        ))}
      </div>

      {history && history.length > 1 && (
        <div className="panel">
          <div className="panel-title">Golden Record history ({history.length} versions)</div>
          {[...history].reverse().map(v => (
            <div key={v.version_number} className="evidence-drawer" style={{ marginBottom: 6 }}>
              <b>v{v.version_number}</b> &middot; {v.reason} &middot; by {v.created_by} &middot; {v.created_at}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function FieldRow({ field, label, value, lineage, conflicts, confidence, needsReview, onResolve, onReject, onEdit }) {
  const [open, setOpen] = useState(false);
  const [editVal, setEditVal] = useState('');
  const hasConflicts = conflicts && conflicts.length > 0;
  const unresolved = lineage?.rule === 'UNRESOLVED_CONFLICT';
  return (
    <div className={`field-card gr-field${needsReview ? ' review' : ''}`}>
      <div className="field-card-head">
        <span className="field-name">{label || field}</span>
        <div className="btn-row">
          {confidence != null && <span className="badge badge-neutral">confidence {confidence.toFixed(2)}</span>}
          {hasConflicts && (
            <button className="btn btn-sm" onClick={() => setOpen(!open)}>{open ? 'Hide' : 'Show'} {conflicts.length} conflict(s)</button>
          )}
        </div>
      </div>
      {needsReview && <div className="warning-item" style={{ marginTop: 8 }}>Conflicting {label || field} values require review.</div>}
      <div className="gr-value">{value == null ? <i>missing</i> : String(value)}</div>
      {lineage && (
        <div className="gr-lineage">
          <span>selected by</span>
          <span className={`rule-chip${unresolved ? ' warn' : ''}`}>{lineage.rule}</span>
          {lineage.source_system && <span className="src-chip">{lineage.source_system}</span>}
          {lineage.source_updated_at && <span className="time-chip" title="When the winning source record was last updated">updated {lineage.source_updated_at}</span>}
          <span>from {lineage.dataset_id || 'steward edit'}{lineage.record_id != null ? ` / record ${lineage.record_id}` : ''}</span>
        </div>
      )}
      {open && hasConflicts && (
        <div className="evidence-drawer" style={{ marginTop: 10 }}>
          {conflicts.map((c, i) => (
            <div key={i} className="conflict-row">
              <div className="who">
                <span className="val">{String(c.value)}</span>
                <span className="src">
                  {c.source_system && <span className="src-chip">{c.source_system}</span>}
                  {c.source_updated_at && <span className="time-chip">updated {c.source_updated_at}</span>}
                  <span>{c.dataset_id}/{c.record_id} &middot; support {c.support}</span>
                </span>
              </div>
              <span className="btn-row">
                <button className="btn btn-sm" onClick={() => onResolve(c.value)}>Use this value</button>
                <button className="btn btn-sm" onClick={() => onReject(c.value)}>Reject value</button>
              </span>
            </div>
          ))}
        </div>
      )}
      <div className="gr-edit">
        <input placeholder="edit value…" value={editVal} onChange={e => setEditVal(e.target.value)} />
        <button className="btn btn-sm" disabled={!editVal} onClick={() => { onEdit(editVal); setEditVal(''); }}>Edit field</button>
      </div>
    </div>
  );
}
