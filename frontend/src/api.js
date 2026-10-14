const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8420';

export class DuplicateFileError extends Error {
  constructor(payload) {
    super(payload.message || 'Duplicate file detected');
    this.name = 'DuplicateFileError';
    this.sha256 = payload.sha256;
    this.existingDatasets = payload.existing_datasets || [];
  }
}

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail;
    } catch (e) { /* non-JSON error body, fall back to statusText */ }

    if (res.status === 409 && detail && detail.error === 'duplicate_file') {
      throw new DuplicateFileError(detail);
    }
    if (detail && detail.error === 'replace_not_supported') {
      throw new Error(detail.message);
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export const api = {
  // paged duplicate groups for a cluster run (used by the Match & Cluster screen)
  duplicateGroups: (datasetId, clusterRunId, offset = 0, limit = 25) =>
    fetch(`${BASE}/api/find-duplicates/${datasetId}/${clusterRunId}/groups?offset=${offset}&limit=${limit}`).then(handle),
  activeModel: (domain) => fetch(`${BASE}/api/domains/${domain}/active-model`).then(handle),
  listDomains: () => fetch(`${BASE}/api/domains`).then(handle),
  domainSchema: (domain) => fetch(`${BASE}/api/domains/${domain}/schema`).then(handle),
  listDatasets: () => fetch(`${BASE}/api/datasets`).then(handle),
  getDataset: (id) => fetch(`${BASE}/api/datasets/${id}`).then(handle),
  deleteDataset: (id) => fetch(`${BASE}/api/datasets/${id}`, { method: 'DELETE' }).then(handle),

  // on_duplicate: 'ask' (default) | 'use_existing' | 'allow_duplicate' | 'replace'
  upload: (domain, file, onDuplicate = 'ask') => {
    const fd = new FormData();
    fd.append('domain', domain);
    fd.append('file', file);
    fd.append('on_duplicate', onDuplicate);
    return fetch(`${BASE}/api/datasets/upload`, { method: 'POST', body: fd }).then(handle);
  },

  mappingSuggestions: (id) => fetch(`${BASE}/api/datasets/${id}/mapping/suggestions`).then(handle),
  getMapping: (id) => fetch(`${BASE}/api/datasets/${id}/mapping`).then(handle),
  confirmMapping: (id, mappings) => fetch(`${BASE}/api/datasets/${id}/mapping`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mappings }),
  }).then(handle),

  runProfile: (id) => fetch(`${BASE}/api/datasets/${id}/profile`, { method: 'POST' }).then(handle),
  getProfile: (id, stage = 'RAW') => fetch(`${BASE}/api/datasets/${id}/profile?stage=${stage}`).then(handle),

  runStandardize: (id) => fetch(`${BASE}/api/datasets/${id}/standardize`, { method: 'POST' }).then(handle),
  runReprofile: (id) => fetch(`${BASE}/api/datasets/${id}/reprofile`, { method: 'POST' }).then(handle),
  qualityComparison: (id) => fetch(`${BASE}/api/datasets/${id}/quality-comparison`).then(handle),

  erConfigSuggestions: (id) => fetch(`${BASE}/api/datasets/${id}/er-config/suggestions`).then(handle),
  submitErConfig: (id, assignments) => fetch(`${BASE}/api/datasets/${id}/er-config`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ assignments }),
  }).then(handle),

  readiness: (id) => fetch(`${BASE}/api/datasets/${id}/readiness`).then(handle),

  // ---------------- Labeling (human-in-the-loop) ----------------
  generateCandidates: (id, n = 20, strategy = 'AUTO', seed = 42) => fetch(`${BASE}/api/datasets/${id}/er/labeling/candidates`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ n, strategy, seed }),
  }).then(handle),
  listLabelPairs: (id, status = 'PENDING', limit = 100) =>
    fetch(`${BASE}/api/datasets/${id}/er/labeling/pairs?status=${status}&limit=${limit}`).then(handle),
  labelPair: (pairId, label, labeler, reason) => fetch(`${BASE}/api/labeling/pairs/${pairId}/label`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label, labeler, reason }),
  }).then(handle),
  addManualLabel: (id, recordIdA, recordIdB, label, labeler, reason) =>
    fetch(`${BASE}/api/datasets/${id}/er/labeling/manual`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ record_id_a: recordIdA, record_id_b: recordIdB, label, labeler, reason }),
    }).then(handle),
  labelsSummary: (domain) => fetch(`${BASE}/api/domains/${domain}/labels/summary`).then(handle),
  // what a training run would take from the label repository (counts, exclusions, split)
  trainingLabelReadiness: (id, seed = 42) =>
    fetch(`${BASE}/api/datasets/${id}/er/labeling/training-readiness?seed=${seed}`).then(handle),
  labelledPairs: (domain, { label, origin, offset = 0, limit = 25 } = {}) => {
    const p = new URLSearchParams({ offset, limit });
    if (label) p.set('label', label);
    if (origin) p.set('origin', origin);
    return fetch(`${BASE}/api/domains/${domain}/labels/pairs?${p}`).then(handle);
  },
  domainLabels: (domain, label) => fetch(`${BASE}/api/domains/${domain}/labels${label ? `?label=${label}` : ''}`).then(handle),

  // ---------------- Training / Model registry ----------------
  runTraining: (id, opts = {}) => fetch(`${BASE}/api/datasets/${id}/er/training`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seed: 42, synthetic_mode: 'AUTO', use_real_labels: true, ...opts }),
  }).then(handle),
  getTrainingRun: (id, trainingRunId) => fetch(`${BASE}/api/datasets/${id}/er/training/${trainingRunId}`).then(handle),
  listModels: (domain) => fetch(`${BASE}/api/models?domain=${domain}`).then(handle),
  getModel: (runId) => fetch(`${BASE}/api/models/${runId}`).then(handle),
  validateModel: (runId) => fetch(`${BASE}/api/models/${runId}/validate`, { method: 'POST' }).then(handle),
  activateModel: (runId) => fetch(`${BASE}/api/models/${runId}/activate`, { method: 'POST' }).then(handle),
  retireModel: (runId) => fetch(`${BASE}/api/models/${runId}/retire`, { method: 'POST' }).then(handle),
  evaluateModelReal: (runId) => fetch(`${BASE}/api/models/${runId}/evaluate-real`, { method: 'POST' }).then(handle),
  activeModel: (domain) => fetch(`${BASE}/api/domains/${domain}/active-model`).then(handle),

  // ---------------- Matching / Clustering ----------------
  runMatch: (id, opts = {}) => fetch(`${BASE}/api/datasets/${id}/er/match`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(opts),
  }).then(handle),
  getMatchRun: (id, matchRunId) => fetch(`${BASE}/api/datasets/${id}/er/matches/${matchRunId}`).then(handle),
  runCluster: (id, matchRunId, threshold) => fetch(`${BASE}/api/datasets/${id}/er/cluster`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ match_run_id: matchRunId, ...(threshold != null ? { threshold } : {}) }),
  }).then(handle),
  getClusterRun: (id, clusterRunId) => fetch(`${BASE}/api/datasets/${id}/er/clusters/${clusterRunId}`).then(handle),
  generateGoldenRecords: (id, clusterRunId) =>
    fetch(`${BASE}/api/datasets/${id}/er/clusters/${clusterRunId}/golden-records`, { method: 'POST' }).then(handle),

  // ---------------- Entities / Golden Records ----------------
  listEntities: (domain, opts = {}) => {
    const p = new URLSearchParams({ domain, ...opts });
    return fetch(`${BASE}/api/entities?${p}`).then(handle);
  },
  getEntity: (entityId) => fetch(`${BASE}/api/entities/${entityId}`).then(handle),
  getGoldenRecord: (entityId) => fetch(`${BASE}/api/entities/${entityId}/golden-record`).then(handle),
  getGoldenRecordHistory: (entityId) => fetch(`${BASE}/api/entities/${entityId}/golden-record/history`).then(handle),
  getReviewContext: (entityId) => fetch(`${BASE}/api/entities/${entityId}/review-context`).then(handle),
  submitReview: (entityId, payload) => fetch(`${BASE}/api/entities/${entityId}/review`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }).then(handle),

  // ---------------- Steward Review queue ----------------
  pendingReviews: (domain, offset = 0, limit = 25) =>
    fetch(`${BASE}/api/reviews/pending?domain=${domain}&offset=${offset}&limit=${limit}`).then(handle),
  entitiesNeedingReview: (domain, offset = 0, limit = 25) =>
    fetch(`${BASE}/api/reviews/entities-needing-review?domain=${domain}&offset=${offset}&limit=${limit}`).then(handle),
  resolvePendingReview: (pendingReviewId, decision, reviewer, reason) =>
    fetch(`${BASE}/api/reviews/pending/${pendingReviewId}/resolve`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision, reviewer, reason }),
    }).then(handle),

  // ---------------- Survivorship configuration ----------------
  getSurvivorship: (domain) => fetch(`${BASE}/api/domains/${domain}/survivorship`).then(handle),
  putSurvivorship: (domain, payload) => fetch(`${BASE}/api/domains/${domain}/survivorship`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }).then(handle),
  detectedSurvivorship: (domain) => fetch(`${BASE}/api/domains/${domain}/survivorship/detected`).then(handle),
  submitSurvivorship: (domain, updatedBy = 'steward_ui') =>
    fetch(`${BASE}/api/domains/${domain}/survivorship/submit?updated_by=${encodeURIComponent(updatedBy)}`, { method: 'POST' }).then(handle),
  confirmSurvivorship: (domain, updatedBy = 'steward_ui') =>
    fetch(`${BASE}/api/domains/${domain}/survivorship/confirm?updated_by=${encodeURIComponent(updatedBy)}`, { method: 'POST' }).then(handle),

  // ---------------- Final master data (approved Golden Records) ----------------
  // scope: 'final' | 'awaiting'; kind: 'all' | 'merged' | 'distinct'; q: text search
  masterData: (domain, offset = 0, limit = 50, { scope = 'final', kind = 'all', q = '' } = {}) => {
    const p = new URLSearchParams({ offset, limit, scope, kind });
    if (q && q.trim()) p.set('q', q.trim());
    return fetch(`${BASE}/api/domains/${domain}/master-data?${p}`).then(handle);
  },

  // ---------------- Continuous incoming data ----------------
  applyActiveModel: (id, opts = {}) => fetch(`${BASE}/api/datasets/${id}/er/apply-active-model`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(opts),
  }).then(handle),
  submitIncomingRecords: (domain, records, opts = {}) => fetch(`${BASE}/api/domains/${domain}/incoming`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ records, source_name: 'ui_manual_entry', ...opts }),
  }).then(handle),
  submitIncomingFile: (domain, file, opts = {}) => {
    const fd = new FormData();
    fd.append('file', file);
    if (opts.match_threshold != null) fd.append('match_threshold', opts.match_threshold);
    if (opts.ambiguous_threshold != null) fd.append('ambiguous_threshold', opts.ambiguous_threshold);
    return fetch(`${BASE}/api/domains/${domain}/incoming/file`, { method: 'POST', body: fd }).then(handle);
  },
  listIncomingBatches: (domain) => fetch(`${BASE}/api/domains/${domain}/incoming-batches`).then(handle),
  getIncomingBatchResults: (id, batchRunId) =>
    fetch(`${BASE}/api/datasets/${id}/er/incoming-batches/${batchRunId}/results`).then(handle),
  incomingAudit: (domain, decision) =>
    fetch(`${BASE}/api/domains/${domain}/incoming-audit${decision ? `?decision=${decision}` : ''}`).then(handle),

  // ---------------- Jobs ----------------
  listJobs: (opts = {}) => {
    const p = new URLSearchParams(opts);
    return fetch(`${BASE}/api/jobs?${p}`).then(handle);
  },
};
