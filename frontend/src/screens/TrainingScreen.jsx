import React, { useEffect, useState, useCallback } from 'react';
import { api } from '../api';

// Training screen owns the model lifecycle: TRAIN -> DRAFT -> Validate ->
// VALIDATED -> Activate -> ACTIVE -> (only then) proceed to Matching.
// It NEVER auto-navigates to Matching: a freshly trained model is DRAFT and
// matching with an unvalidated model is exactly the bug this prevents.
export default function TrainingScreen({ dataset, onContinueToMatching, onLabelMore, datasetReady = true }) {
  const [syntheticMode, setSyntheticMode] = useState('AUTO');
  const [useRealLabels, setUseRealLabels] = useState(true);
  const [threshold, setThreshold] = useState('');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [models, setModels] = useState([]);
  const [labelInfo, setLabelInfo] = useState(null);      // what the label repository offers this training run
  const [labelInfoError, setLabelInfoError] = useState(null);

  useEffect(() => {
    let live = true;
    setLabelInfo(null); setLabelInfoError(null);
    api.trainingLabelReadiness(dataset.dataset_id)
      .then(r => { if (live) setLabelInfo(r); })
      .catch(e => { if (live) setLabelInfoError(e.message); });
    return () => { live = false; };
  }, [dataset.dataset_id, result?.training_run_id]);

  const refreshModels = useCallback(() => {
    api.listModels(dataset.domain).then(r => setModels(r.models)).catch(() => {});
  }, [dataset.domain]);

  useEffect(() => { refreshModels(); }, [refreshModels]);

  async function train() {
    setRunning(true); setError(null); setResult(null);
    try {
      const opts = { synthetic_mode: syntheticMode, use_real_labels: useRealLabels };
      if (threshold !== '') opts.threshold = parseFloat(threshold);
      const r = await api.runTraining(dataset.dataset_id, opts);
      setResult(r);
      refreshModels();
      // Deliberately NO navigation here: the model is DRAFT until explicitly
      // validated and activated (see ModelCard).
    } catch (e) { setError(e.message); } finally { setRunning(false); }
  }

  async function lifecycle(runId, action) {
    setError(null);
    try {
      const fn = action === 'validate' ? api.validateModel : action === 'activate' ? api.activateModel : api.retireModel;
      await fn(runId);
      refreshModels();
    } catch (e) { setError(e.message); }
  }

  const activeModel = models.find(m => m.lifecycle_status === 'ACTIVE');

  return (
    <div>
      <div className="page-header">
        <h1>Training &amp; model lifecycle</h1>
        <p>
          Real steward labels are preferred evidence; additional supporting training data is used only
          when needed. A trained model starts as <b>DRAFT</b> and must be explicitly
          <b> validated</b> and <b> activated</b> before matching can use it.
        </p>
      </div>

      {error && <div className="status-banner error">{error}</div>}
      {!datasetReady && (
        <div className="status-banner error">
          This dataset is not READY_FOR_DEDUPE yet — complete ER configuration and readiness first.
        </div>
      )}

      <LabelRepositoryPanel info={labelInfo} error={labelInfoError} domain={dataset.domain}
                            useRealLabels={useRealLabels} syntheticMode={syntheticMode} onLabelMore={onLabelMore} />

      <div className="panel">
        <div className="panel-title">Run training</div>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <label style={{ fontSize: 12.5 }}>
            Additional training data<br />
            <select value={syntheticMode} onChange={e => setSyntheticMode(e.target.value)} style={{ padding: 6, marginTop: 4 }}>
              <option value="AUTO">Auto (only if real labels are too few)</option>
              <option value="ALWAYS">Always</option>
              <option value="NEVER">Never (real labels only)</option>
            </select>
          </label>
          <label style={{ fontSize: 12.5, display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={useRealLabels} onChange={e => setUseRealLabels(e.target.checked)} />
            Use real steward labels
          </label>
          <label style={{ fontSize: 12.5 }}>
            Threshold override<br />
            <input placeholder="auto-calibrate" value={threshold} onChange={e => setThreshold(e.target.value)}
                   style={{ width: 110, padding: 6, marginTop: 4 }} />
          </label>
          <button className="btn btn-primary" disabled={running || !datasetReady} onClick={train}>
            {running ? 'Training…' : 'Train model'}
          </button>
        </div>
      </div>

      {result && <TrainingResultPanel result={result} />}

      <div className="panel-title" style={{ marginTop: 8 }}>Model registry &middot; {dataset.domain}</div>
      {models.length === 0 && <div className="spinner-text">No models trained yet for this domain.</div>}
      {models.map(m => <ModelCard key={m.training_run_id} model={m} onAction={lifecycle} />)}
      {activeModel && datasetReady && (
        <div className="panel" style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ fontWeight: 700 }}>✓ Model {activeModel.training_run_id} is ACTIVE and ready for matching.</span>
          {onContinueToMatching && (
            <button className="btn btn-primary" data-testid="continue-to-matching" onClick={onContinueToMatching}>
              Continue to Matching →
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function TrainingResultPanel({ result: r }) {
  if (r.status !== 'TRAINED') {
    return (
      <div className="panel">
        <div className="panel-title">Training run &middot; <span className="badge badge-bad">{r.status}</span></div>
        <div className="evidence-drawer">{r.error}</div>
      </div>
    );
  }
  return (
    <div className="panel">
      <div className="panel-title">Training run &middot; <span className="badge badge-good">TRAINED</span></div>
      <div className="evidence-drawer" style={{ marginBottom: 10 }}>
        Model registered as <b>DRAFT</b> — validate it below, then activate. Matching stays locked until the model is ACTIVE.
      </div>
      {r.label_counts && (
        <div className="evidence-drawer" style={{ marginBottom: 10 }}>
          <b>{r.label_counts.real_steward.match}</b> real MATCH / <b>{r.label_counts.real_steward.distinct}</b> real DISTINCT labels
          {r.label_counts.real_steward.train && (
            <> ({r.label_counts.real_steward.train.match} / {r.label_counts.real_steward.train.distinct} in the training split)</>
          )}
          &nbsp;&middot;&nbsp; additional training data {r.label_counts.synthetic.used ? 'used' : 'not used'} ({r.label_counts.synthetic.mode})
          {r.label_counts.excluded_missing_match_fields > 0 && (
            <>&nbsp;&middot;&nbsp; {r.label_counts.excluded_missing_match_fields} label(s) skipped (missing MATCH fields)</>
          )}
        </div>
      )}
      <div className="two-col">
        <EvalBlock title="Supplementary validation" note="Additional held-out evaluation examples used to verify the pipeline." metrics={r.evaluation_metrics} />
        <EvalBlock title="Real steward-labelled evaluation" note={r.real_evaluation?.note} metrics={r.real_evaluation?.validation} extra={r.real_evaluation} />
      </div>
      <div className="metric-row" style={{ marginTop: 12 }}>
        <div className="metric">threshold <b>{r.threshold}</b></div>
        <div className="metric">calibration <b>{r.threshold_calibration}</b></div>
        <div className="metric">review band from <b>{r.review_threshold}</b></div>
      </div>
    </div>
  );
}

function EvalBlock({ title, note, metrics, extra }) {
  if (!metrics || metrics.pair_count === 0 && !metrics.validation_pair_count) {
    return (
      <div className="evidence-drawer">
        <div style={{ fontWeight: 700, marginBottom: 4 }}>{title}</div>
        No evidence available for this evaluation yet.
      </div>
    );
  }
  const count = metrics.validation_pair_count ?? metrics.pair_count;
  return (
    <div className="evidence-drawer">
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 11, color: 'var(--slate)', marginBottom: 8 }}>{note}</div>
      <div className="metric-row">
        <div className="metric">pairs <b>{count}</b></div>
        <div className="metric">precision <b>{fmt(metrics.precision)}</b></div>
        <div className="metric">recall <b>{fmt(metrics.recall)}</b></div>
        <div className="metric">F1 <b>{fmt(metrics.f1)}</b></div>
      </div>
      <div className="metric-row">
        <div className="metric">TP <b>{metrics.true_positives}</b></div>
        <div className="metric">FP <b>{metrics.false_positives}</b></div>
        <div className="metric">FN <b>{metrics.false_negatives}</b></div>
        <div className="metric">TN <b>{metrics.true_negatives}</b></div>
      </div>
      {extra?.test?.pair_count > 0 && (
        <div className="metric-row"><div className="metric">held-out test set: <b>{extra.test.pair_count}</b> pairs, F1 <b>{fmt(extra.test.f1)}</b></div></div>
      )}
    </div>
  );
}

function fmt(v) { return v == null ? '—' : v.toFixed(3); }

function ModelCard({ model: m, onAction }) {
  const badgeClass = { DRAFT: 'badge-neutral', VALIDATED: 'badge-warn', ACTIVE: 'badge-good', RETIRED: 'badge-bad' }[m.lifecycle_status] || 'badge-neutral';
  return (
    <div className="field-card" data-testid={`model-card-${m.training_run_id}`}>
      <div className="field-card-head">
        <span className="field-name">{m.training_run_id}</span>
        <span className={`badge ${badgeClass}`} data-testid="model-lifecycle">{m.lifecycle_status}</span>
      </div>
      <div className="metric-row">
        <div className="metric">match fields <b>{m.match_fields.join(', ')}</b></div>
        <div className="metric">threshold <b>{m.threshold}</b> ({m.threshold_calibration})</div>
        {m.is_active && <span className="badge badge-good">CURRENTLY ACTIVE</span>}
      </div>
      {m.compatibility_problems?.length > 0 && (
        <div className="warning-list">
          {m.compatibility_problems.map((p, i) => <div key={i} className="warning-item">{p}</div>)}
        </div>
      )}
      <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
        {m.lifecycle_status === 'DRAFT' && (
          <button className="btn btn-sm btn-primary" data-testid="validate-model" onClick={() => onAction(m.training_run_id, 'validate')}>
            Validate Model
          </button>
        )}
        {m.lifecycle_status === 'VALIDATED' && (
          <button className="btn btn-sm btn-primary" data-testid="activate-model" onClick={() => onAction(m.training_run_id, 'activate')}>
            Activate Model
          </button>
        )}
        {m.lifecycle_status === 'ACTIVE' && (
          <button className="btn btn-sm" onClick={() => onAction(m.training_run_id, 'retire')}>Retire</button>
        )}
      </div>
    </div>
  );
}


// Shows, BEFORE training, what the labelled-pair repository contributes: how many steward labels exist,
// how many this model can use, how they would be split, and what each training option will therefore do.
function LabelRepositoryPanel({ info, error, domain, useRealLabels, syntheticMode, onLabelMore }) {
  if (error) {
    return <div className="panel"><div className="panel-title">Labelled pairs · {domain}</div>
      <div className="evidence-drawer">Could not read the label repository: {error}</div></div>;
  }
  if (!info) return <div className="panel"><div className="panel-title">Labelled pairs · {domain}</div><div className="spinner-text">Loading…</div></div>;
  const { repository: r, usable: u, split_preview: sp, verdict: v } = info;
  let willDo;
  if (!useRealLabels) willDo = 'Real steward labels are switched off: training uses synthetic pairs only.';
  else if (syntheticMode === 'NEVER') willDo = v.real_only_training_possible
    ? 'Training will use ONLY your real labels.'
    : 'Not enough real labels for "Never" - training would stop with INSUFFICIENT_TRAINING_DATA.';
  else if (syntheticMode === 'ALWAYS') willDo = 'Training will use your real labels plus synthetic pairs.';
  else willDo = v.real_labels_sufficient
    ? 'Auto: your real labels are enough, so no synthetic pairs are added.'
    : 'Auto: your real labels are few, so synthetic pairs are added next to them.';
  return (
    <div className="panel" data-testid="label-repository-panel">
      <div className="panel-title">Labelled pairs available for training · {domain}</div>
      <div className="metric-row">
        <div className="metric"><b>{r.match}</b> MATCH</div>
        <div className="metric"><b>{r.distinct}</b> DISTINCT</div>
        <div className="metric"><b>{r.unsure_excluded}</b> UNSURE (never trained on)</div>
        <div className="metric"><b>{r.awaiting_label}</b> not labelled yet</div>
      </div>
      <div className="metric-row">
        <div className="metric">usable by this model <b>{u.total}</b> ({u.match} MATCH / {u.distinct} DISTINCT)</div>
        <div className="metric">from Steward Review <b>{u.origin.steward_review}</b></div>
        <div className="metric">from Labeling screen <b>{u.origin.labeling_screen}</b></div>
      </div>
      {u.total > 0 && (
        <div className="metric-row">
          <div className="metric">train <b>{sp.train.match}/{sp.train.distinct}</b></div>
          <div className="metric">validation <b>{sp.validation.match}/{sp.validation.distinct}</b></div>
          <div className="metric">test <b>{sp.test.match}/{sp.test.distinct}</b></div>
          <div className="metric" style={{ color: 'var(--slate)' }}>MATCH/DISTINCT per split; records never appear in two splits</div>
        </div>
      )}
      <div className="evidence-drawer" style={{ marginTop: 8 }}><b>{willDo}</b></div>
      {info.notes?.map((n, i) => <div key={i} className="warning-item" style={{ marginTop: 6 }}>{n}</div>)}
      <div className="btn-row" style={{ marginTop: 8 }}>
        {onLabelMore && <button className="btn btn-sm" onClick={onLabelMore}>Label more pairs</button>}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--slate)', marginTop: 6 }}>
        Steward Review decisions (Match / Distinct) are saved here automatically. Nothing retrains by itself:
        a new model is only built when you click “Train model”.
      </div>
    </div>
  );
}
