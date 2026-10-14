import React, { useEffect, useState, useCallback } from 'react';
import { api } from '../api';
import { useDomainSchema } from '../fieldOrder';
import RecordFields from '../components/RecordFields';

// The steward is confirming whether two records describe the SAME real-world
// entity -- never asked to review random pairs; candidates come from Dedupe's
// own active learner (bootstrap) or, once a model exists, its own scoring
// (uncertain / strong / hard-negative bands).
export default function LabelingScreen({ dataset, onContinueToTraining }) {
  const schema = useDomainSchema(dataset.domain);
  const [pending, setPending] = useState([]);
  const [summary, setSummary] = useState(null);
  const [n, setN] = useState(15);
  const [strategy, setStrategy] = useState('AUTO');
  const [labeler, setLabeler] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [lastBatch, setLastBatch] = useState(null);

  const refresh = useCallback(() => {
    api.listLabelPairs(dataset.dataset_id, 'PENDING', 100).then(r => setPending(r.pairs)).catch(e => setError(e.message));
    api.labelsSummary(dataset.domain).then(setSummary).catch(() => {});
  }, [dataset.dataset_id, dataset.domain]);

  useEffect(() => { refresh(); }, [refresh]);

  async function generate() {
    setBusy(true); setError(null);
    try {
      const r = await api.generateCandidates(dataset.dataset_id, n, strategy);
      setLastBatch(r);
      refresh();
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  async function label(pairId, value) {
    if (!labeler.trim()) { setError('Enter your name/id as the labeler first.'); return; }
    setError(null);
    try {
      await api.labelPair(pairId, value, labeler.trim());
      setPending(prev => prev.filter(p => p.pair_id !== pairId));
      api.labelsSummary(dataset.domain).then(setSummary);
    } catch (e) { setError(e.message); }
  }

  return (
    <div>
      <div className="page-header">
        <h1>Labeling</h1>
        <p>
          For each pair below, confirm whether <b>Record A</b> and <b>Record B</b> represent the
          <b> same real-world {dataset.domain}</b>. Candidates come from Dedupe's own candidate
          generation, not a random sample &mdash; the number needed depends on this dataset, not a fixed rule.
        </p>
      </div>

      {error && <div className="status-banner error">{error}</div>}

      {summary && (
        <div className="panel">
          <div className="panel-title">Label repository &middot; {dataset.domain}</div>
          <div className="metric-row">
            <div className="metric"><b>{summary.MATCH}</b> MATCH</div>
            <div className="metric"><b>{summary.DISTINCT}</b> DISTINCT</div>
            <div className="metric"><b>{summary.UNSURE}</b> UNSURE (excluded from training)</div>
            <div className="metric"><b>{summary.usable_for_training}</b> usable for training</div>
          </div>
          <div className="evidence-drawer" style={{ marginTop: 10 }}>{summary.note}</div>
        </div>
      )}

      <div className="panel">
        <div className="panel-title">What each label means</div>
        <div className="two-col">
          <div className="evidence-drawer">
            <b>MATCH</b> — both records represent the <i>same</i> real-world entity.<br />
            <b>DISTINCT</b> — the records represent <i>different</i> real-world entities.<br />
            <b>UNSURE</b> — insufficient evidence to decide (kept out of training).
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">Generate candidate pairs</div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <label style={{ fontSize: 12.5 }}>
            How many<br />
            <input type="number" min={1} max={200} value={n} onChange={e => setN(+e.target.value)}
                   style={{ width: 70, padding: 6, marginTop: 4 }} />
          </label>
          <label style={{ fontSize: 12.5 }}>
            Strategy<br />
            <select value={strategy} onChange={e => setStrategy(e.target.value)} style={{ padding: 6, marginTop: 4 }}>
              <option value="AUTO">Auto (model-assisted if one is active)</option>
              <option value="BOOTSTRAP">Bootstrap (Dedupe active learning, no model needed)</option>
              <option value="MODEL">Model-assisted (requires an active model)</option>
            </select>
          </label>
          <button className="btn btn-primary" disabled={busy} onClick={generate}>
            {busy ? 'Generating…' : 'Generate candidates'}
          </button>
        </div>
        {lastBatch && (
          <div className="evidence-drawer" style={{ marginTop: 12 }}>
            Created {lastBatch.created} candidate pair(s) via <b>{lastBatch.strategy_used}</b>. {lastBatch.instruction}
          </div>
        )}
      </div>

      <div className="panel">
        <div className="panel-title">Your identity (recorded with every label)</div>
        <input placeholder="e.g. priya.s" value={labeler} onChange={e => setLabeler(e.target.value)}
               className="field-input" />
      </div>

      <div className="panel-title" style={{ marginTop: 4 }}>Pending pairs ({pending.length})</div>
      {pending.length === 0 && <div className="spinner-text">No pending pairs. Generate some above.</div>}
      {pending.map(p => <PairCard key={p.pair_id} pair={p} onLabel={label} schema={schema} />)}

      <LabelRepositoryBrowser domain={dataset.domain} schema={schema} refreshKey={summary?.total} />

      <div className="panel" style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 8 }}>
        <span style={{ fontSize: 12.5, color: 'var(--slate)' }}>
          Labeled enough examples for now? Proceed to training with your real steward labels.
        </span>
        <button className="btn btn-primary" data-testid="continue-to-training"
                disabled={!summary || !(summary.usable_for_training > 0)}
                onClick={onContinueToTraining}>
          Continue to Training →
        </button>
      </div>
    </div>
  );
}

function PairCard({ pair, onLabel, schema }) {
  const ev = pair.evidence?.identifier_agreement || {};
  return (
    <div className="field-card">
      <div className="field-card-head">
        <span className="badge badge-neutral">{pair.strategy}</span>
        {pair.model_score != null && (
          <span className="badge badge-warn">model score {pair.model_score.toFixed(3)}</span>
        )}
      </div>
      <div className="two-col" style={{ marginTop: 10 }}>
        <RecordBlock label="Record A" values={pair.display_a || pair.values_a} schema={schema} />
        <RecordBlock label="Record B" values={pair.display_b || pair.values_b} schema={schema} />
      </div>
      {Object.keys(ev).length > 0 && (
        <div style={{ marginTop: 8 }}>
          {Object.entries(ev).map(([f, v]) => (
            <span key={f} className="signal-chip">{f}: {v.toLowerCase()}</span>
          ))}
          <div className="evidence-drawer" style={{ marginTop: 6, fontSize: 11 }}>
            Identifier agreement is a hint only &mdash; it never sets the label.
          </div>
        </div>
      )}
      <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
        <button className="btn btn-sm" style={{ borderColor: 'var(--good)', color: 'var(--good)' }}
                onClick={() => onLabel(pair.pair_id, 'MATCH')}>Same entity (MATCH)</button>
        <button className="btn btn-sm" style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}
                onClick={() => onLabel(pair.pair_id, 'DISTINCT')}>Different (DISTINCT)</button>
        <button className="btn btn-sm" onClick={() => onLabel(pair.pair_id, 'UNSURE')}>Unsure</button>
      </div>
    </div>
  );
}

function RecordBlock({ label, values, schema }) {
  return (
    <div className="evidence-drawer">
      <div style={{ fontWeight: 700, marginBottom: 6 }}>{label}</div>
      <RecordFields values={values} schema={schema} showMissing />
    </div>
  );
}

const ORIGIN_TEXT = { REVIEW_DECISION: 'Steward Review' };
const LABEL_BADGE = { MATCH: 'badge-good', DISTINCT: 'badge-bad', UNSURE: 'badge-neutral' };

function nameOf(values) {
  const k = Object.keys(values || {}).find(x => x.endsWith('_name') && values[x]);
  return (k && values[k]) || (values && Object.values(values).find(v => v)) || '(no values)';
}

// Every labelled pair of the domain -- including the ones created by Steward Review decisions -- with the
// label, who made it and where it came from. These are the pairs a training run draws on.
function LabelRepositoryBrowser({ domain, schema, refreshKey }) {
  const [open, setOpen] = useState(false);
  const [origin, setOrigin] = useState('');
  const [label, setLabel] = useState('');
  const [data, setData] = useState({ pairs: [], total: 0, has_more: false });
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [expanded, setExpanded] = useState({});

  const load = useCallback(async (offset, replace) => {
    setLoading(true); setErr(null);
    try {
      const r = await api.labelledPairs(domain, { origin: origin || undefined, label: label || undefined, offset, limit: 25 });
      setData(prev => ({ ...r, pairs: replace ? r.pairs : [...prev.pairs, ...r.pairs] }));
    } catch (e) { setErr(e.message); } finally { setLoading(false); }
  }, [domain, origin, label]);

  useEffect(() => { if (open) load(0, true); }, [open, load, refreshKey]);

  return (
    <div className="panel">
      <div className="panel-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Labelled pairs in the repository</span>
        <button className="btn btn-sm" onClick={() => setOpen(o => !o)}>{open ? 'Hide' : 'Show'}</button>
      </div>
      {open && (
        <>
          <div className="btn-row" style={{ marginBottom: 8 }}>
            <select aria-label="Origin" value={origin} onChange={e => setOrigin(e.target.value)} style={{ padding: 6, borderRadius: 4, fontSize: 12.5 }}>
              <option value="">All origins</option>
              <option value="review">Steward Review decisions</option>
              <option value="labeling">Labeling screen</option>
            </select>
            <select aria-label="Label" value={label} onChange={e => setLabel(e.target.value)} style={{ padding: 6, borderRadius: 4, fontSize: 12.5 }}>
              <option value="">All labels</option>
              <option value="MATCH">MATCH</option>
              <option value="DISTINCT">DISTINCT</option>
              <option value="UNSURE">UNSURE</option>
            </select>
            <span className="metric"><b>{data.total}</b> pair(s)</span>
          </div>
          {err && <div className="status-banner error">{err}</div>}
          {!loading && data.pairs.length === 0 && !err && <div className="spinner-text">No labelled pairs for this filter.</div>}
          {data.pairs.map(p => (
            <div key={p.pair_id} className="field-card">
              <div className="field-card-head">
                <span className="field-name" style={{ fontSize: 12.5 }}>{String(nameOf(p.values_a))} ↔ {String(nameOf(p.values_b))}</span>
                <div>
                  <span className={`badge ${LABEL_BADGE[p.label] || 'badge-neutral'}`}>{p.label}</span>
                  <span className="badge badge-neutral" style={{ marginLeft: 6 }}>{ORIGIN_TEXT[p.strategy] || 'Labeling screen'}</span>
                </div>
              </div>
              <div className="metric-row">
                <div className="metric">by <b>{p.labeler || '—'}</b></div>
                {p.model_score != null && <div className="metric">model score <b>{p.model_score.toFixed(3)}</b></div>}
                {p.labeled_at && <div className="metric">{p.labeled_at.slice(0, 16).replace('T', ' ')}</div>}
                <button className="btn btn-sm" onClick={() => setExpanded(x => ({ ...x, [p.pair_id]: !x[p.pair_id] }))}>
                  {expanded[p.pair_id] ? 'Hide records' : 'Show records'}
                </button>
              </div>
              {expanded[p.pair_id] && (
                <div className="two-col" style={{ marginTop: 8 }}>
                  <RecordBlock label="Record A" values={p.display_a || p.values_a} schema={schema} />
                  <RecordBlock label={String(p.record_id_b).startsWith('ENTITY::') ? 'Golden Record B' : 'Record B'}
                               values={p.display_b || p.values_b} schema={schema} />
                </div>
              )}
            </div>
          ))}
          {data.has_more && (
            <button className="btn btn-sm" disabled={loading} onClick={() => load(data.pairs.length, false)}>
              {loading ? 'Loading…' : 'Load 25 more'}
            </button>
          )}
        </>
      )}
    </div>
  );
}
