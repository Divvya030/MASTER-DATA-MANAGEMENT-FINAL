import React, { useEffect, useState } from 'react';
import { api } from '../api';
import ClusterGroupsView from '../components/ClusterGroupsView';

// Match & Cluster is ONE user action. Internally it is still the existing MatchRun followed by the
// existing ClusterRun (clustering starts automatically when matching succeeds). Golden records are
// NOT generated automatically: the user reviews the clusters first, then clicks Generate.
// Matching REQUIRES an ACTIVE model for the domain (the server enforces this with a 422; we pre-check
// so the user sees a friendly gate). Clicking Match never trains anything.
export default function MatchingScreen({ dataset, onGoldenRecords, onStewardReview }) {
  const [threshold, setThreshold] = useState('');
  const [phase, setPhase] = useState('idle'); // idle | matching | clustering | done
  const [matchRun, setMatchRun] = useState(null);
  const [clusterRun, setClusterRun] = useState(null);
  const [failure, setFailure] = useState(null); // { stage: 'matching' | 'clustering' | 'golden', message }
  const [notice, setNotice] = useState(null);
  const [generating, setGenerating] = useState(false);
  const [goldenResult, setGoldenResult] = useState(null);
  const [activeModel, setActiveModel] = useState(undefined); // undefined = checking

  useEffect(() => {
    let cancelled = false;
    api.activeModel(dataset.domain)
      .then(m => { if (!cancelled) setActiveModel(m); })
      .catch(() => { if (!cancelled) setActiveModel(null); });
    return () => { cancelled = true; };
  }, [dataset.domain]);

  const busy = phase === 'matching' || phase === 'clustering';

  async function runMatchAndCluster() {
    setPhase('matching'); setFailure(null); setNotice(null);
    setMatchRun(null); setClusterRun(null); setGoldenResult(null);
    // ---- 1. existing matching API
    let match;
    try {
      const opts = {};
      if (threshold !== '') opts.threshold = parseFloat(threshold);
      match = await api.runMatch(dataset.dataset_id, opts);
    } catch (e) {
      setFailure({ stage: 'matching', message: e.message }); setPhase('idle'); return;
    }
    if (match.status !== 'COMPLETED') {
      setMatchRun(match);
      setFailure({ stage: 'matching', message: match.error || 'Matching did not complete.' });
      setPhase('idle'); return;   // clustering is NOT attempted
    }
    setMatchRun(match);
    if (!match.candidate_pair_count) {
      setNotice('Matching completed: the model proposed no candidate duplicate pairs, so there is nothing to cluster.');
      setPhase('done'); return;
    }
    // ---- 2. existing clustering API, started automatically (uses the match run's threshold)
    setPhase('clustering');
    try {
      const cluster = await api.runCluster(dataset.dataset_id, match.match_run_id);
      if (cluster.status !== 'COMPLETED') {
        setFailure({ stage: 'clustering', message: cluster.error || 'Clustering did not complete.' });
        setPhase('idle'); return;
      }
      setClusterRun(cluster);
      setPhase('done');
    } catch (e) {
      setFailure({ stage: 'clustering', message: e.message }); setPhase('idle');
    }
  }

  async function runGolden() {
    setGenerating(true); setFailure(null);
    try {
      const r = await api.generateGoldenRecords(dataset.dataset_id, clusterRun.cluster_run_id);
      setGoldenResult(r);
      // Navigation is explicit (buttons below), never automatic.
    } catch (e) { setFailure({ stage: 'golden', message: e.message }); } finally { setGenerating(false); }
  }

  if (activeModel === undefined) {
    return <div className="spinner-text">Checking for an ACTIVE model…</div>;
  }

  if (activeModel === null) {
    return (
      <div>
        <div className="page-header"><h1>Match &amp; Cluster</h1></div>
        <div className="panel">
          <div className="panel-title">No ACTIVE model</div>
          <div className="evidence-drawer">
            Matching runs the domain's <b>ACTIVE</b> model. No model is currently active for
            domain <b>{dataset.domain}</b>.
            Go to <b>Train</b>, then <b>Validate</b> and <b>Activate</b> a trained model first.
          </div>
        </div>
      </div>
    );
  }

  const buttonLabel = phase === 'matching' ? 'Matching…' : phase === 'clustering' ? 'Clustering…' : 'Match';
  const be = matchRun?.blocking_evaluation;
  const matchOk = matchRun?.status === 'COMPLETED';
  const failTitle = {
    matching: 'Matching failed',
    clustering: 'Matching completed — Clustering failed',
    golden: 'Golden record generation failed',
  };

  return (
    <div>
      <div className="page-header">
        <h1>Match &amp; Cluster</h1>
        <p>Runs the domain's ACTIVE model against this dataset (no training), then groups the matched records into clusters. Review the clusters, then generate golden records.</p>
      </div>

      {failure && (
        <div className="status-banner error" data-testid="match-failure">
          <b>{failTitle[failure.stage]}</b><br />{failure.message}
        </div>
      )}

      <div className="panel">
        <div className="panel-title">Match</div>
        <div className="evidence-drawer" style={{ marginBottom: 10 }}>
          Active model <b>{activeModel.training_run_id}</b>
          {activeModel.threshold != null && (<> · threshold <b>{activeModel.threshold}</b></>)}
          {activeModel.review_threshold != null && (<> · review band from <b>{activeModel.review_threshold}</b></>)}.
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <label style={{ fontSize: 12.5 }}>
            Threshold override<br />
            <input placeholder="use active model's threshold" value={threshold} onChange={e => setThreshold(e.target.value)}
                   disabled={busy} style={{ width: 190, padding: 6, marginTop: 4 }} />
          </label>
          <button className="btn btn-primary" disabled={busy} onClick={runMatchAndCluster}>{buttonLabel}</button>
          {phase === 'clustering' && <span className="spinner-text">Matching completed. Grouping records into clusters…</span>}
          {phase === 'done' && clusterRun && <span className="metric"><b>Match &amp; Cluster completed</b></span>}
        </div>
        {notice && <div className="status-banner" style={{ marginTop: 12 }}>{notice}</div>}
      </div>

      {matchOk && (
        <div className="panel">
          <div className="panel-title">Match &amp; Cluster result</div>
          <div className="metric-row">
            <div className="metric">records <b>{matchRun.total_records?.toLocaleString()}</b></div>
            <div className="metric">candidate pairs <b>{matchRun.candidate_pair_count?.toLocaleString()}</b></div>
            <div className="metric">MATCH <b>{matchRun.match_count?.toLocaleString()}</b></div>
            <div className="metric">UNSURE <b>{matchRun.uncertain_count?.toLocaleString()}</b></div>
            <div className="metric">NON-MATCH <b>{matchRun.non_match_count?.toLocaleString()}</b></div>
            <div className="metric">threshold <b>{matchRun.threshold}</b></div>
            {clusterRun && (<>
              <div className="metric">clusters <b>{clusterRun.cluster_count?.toLocaleString()}</b></div>
              <div className="metric">singletons <b>{clusterRun.singleton_count?.toLocaleString()}</b></div>
              <div className="metric">largest cluster <b>{clusterRun.largest_cluster_size}</b></div>
            </>)}
          </div>
          {be && (
            <div className="evidence-drawer" style={{ marginTop: 10 }}>
              <div><b>Candidate reduction:</b> {be.candidate_reduction_pct}% of the
                {' '}{be.total_possible_pairs?.toLocaleString()} possible pairs were never scored.
                <i> {be.candidate_reduction_meaning}.</i></div>
              {be.candidate_recall ? (
                <div style={{ marginTop: 6 }}><b>Candidate recall:</b> {(be.candidate_recall.recall * 100).toFixed(1)}%
                  of {be.candidate_recall.known_match_pairs} steward-confirmed MATCH pairs survived blocking.
                  <i> {be.candidate_recall.meaning}.</i></div>
              ) : (
                <div style={{ marginTop: 6 }}>{be.candidate_recall_note}</div>
              )}
            </div>
          )}
          {clusterRun && (
            <div className="evidence-drawer" style={{ marginTop: 10 }}>
              {matchRun.uncertain_count > 0 && (
                <div>UNSURE pairs ({matchRun.uncertain_count?.toLocaleString()}) score in the review band below the threshold. They are <b>not</b> merged into any cluster here — a steward decides each one as MATCH, DISTINCT or UNSURE on the Steward Review screen.</div>
              )}
              <div style={{ marginTop: matchRun.uncertain_count > 0 ? 6 : 0 }}>
                The number of clusters is not claimed to equal the true number of real-world entities unless ground truth exists.
              </div>
            </div>
          )}
        </div>
      )}

      {clusterRun && (
        <div className="panel">
          <div className="panel-title">Clustered records</div>
          <ClusterGroupsView datasetId={dataset.dataset_id} clusterRunId={clusterRun.cluster_run_id} />
        </div>
      )}

      {clusterRun && (
        <div className="panel">
          <div className="panel-title">Golden records</div>
          {!goldenResult && (
            <>
              <div className="metric" style={{ marginBottom: 8 }}>Review the clusters above, then generate golden records from them.</div>
              <button className="btn btn-primary" disabled={generating} onClick={runGolden}>
                {generating ? 'Generating…' : 'Generate Golden Records'}
              </button>
            </>
          )}
          {goldenResult && (
            <div className="evidence-drawer" data-testid="golden-result">
              <div style={{ marginBottom: 8 }}>
                <b>Golden Records Generated</b><br />
                {goldenResult.entities_created?.toLocaleString()} golden entities created; mastered golden records written with full lineage.
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {onGoldenRecords && (
                  <button className="btn btn-sm btn-primary" data-testid="open-entities" onClick={onGoldenRecords}>
                    View Golden Records →
                  </button>
                )}
                {onStewardReview && (
                  <button className="btn btn-sm" data-testid="open-steward-review" onClick={onStewardReview}>
                    Open Steward Review →
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
