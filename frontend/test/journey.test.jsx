/**
 * Frontend journey tests (A–O): the workflow must behave as ONE connected
 * system. Each test drives real screen components with a fully mocked api
 * module (mirroring real backend response shapes).
 *
 *   A. Readiness flip refreshes parent state (no browser refresh needed)
 *   B. Readiness continue CTA navigates to Labeling
 *   C. Readiness not-ready state offers Return-to-ER-Configuration
 *   D. Labeling: generate -> label MATCH/DISTINCT/UNSURE -> persist + summary refresh
 *   E. Labeling: Continue-to-Training gated on usable labels
 *   F. Training: no auto-advance to Matching after TRAINED
 *   G. Training: DRAFT -> Validate -> VALIDATED -> Activate -> ACTIVE
 *   H. Training: Continue-to-Matching only offered when a model is ACTIVE
 *   I. Matching: friendly gate when no ACTIVE model (no raw 422)
 *   J. Matching: ONE Match click -> automatic clustering -> actual clustered records (paged) -> explicit golden step
 *   K. Matching: explicit navigation to Entities / Steward Review after golden
 *   L. Incoming Data: blocked with a clear message when no ACTIVE model
 *   M. Incoming Data: shows active model; submission buttons enabled
 *   N. Pipeline stages unlock only per dataset status (READY_FOR_DEDUPE gates Label/Train/Match)
 *   O. Full app-level flow: readiness continue -> labeling -> training lifecycle
 *   P. Navigation has no Monitoring / Quick-run entries
 *   Q. Final Master Data lists distinct records and pages further results
 *   R. Steward Review loads one page at a time and loads more on demand
 *   S. Incoming Data single-record form shows every canonical field of the domain
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import React from 'react';

const { schemaFor, CUSTOMER_FIELDS } = vi.hoisted(() => {
const CUSTOMER_FIELDS = ['customer_id', 'customer_name', 'legal_name', 'first_name', 'last_name', 'email', 'phone',
  'dob', 'address', 'city', 'state', 'country', 'postal_code', 'website', 'duns', 'tax_identifier', 'registration_number'];
const SUPPLIER_FIELDS = ['supplier_id', 'supplier_name', 'legal_name', 'trade_name', 'email', 'phone', 'address', 'city',
  'state', 'country', 'postal_code', 'website', 'duns', 'tax_identifier', 'registration_number'];
const schemaFor = (domain) => (domain === 'supplier' ? SUPPLIER_FIELDS : CUSTOMER_FIELDS).map(name => ({
  name, label: name.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase()),
  input_type: name === 'email' ? 'email' : name === 'dob' ? 'date' : 'text', identifier: name.endsWith('_id'),
}));
return { schemaFor, CUSTOMER_FIELDS };
});

vi.mock('../src/api.js', () => {
  const dataset = (status) => ({
    dataset_id: 'DS-J-0001', domain: 'customer', row_count: 9996,
    columns: ['customer_name', 'email'], status,
  });
  const readyReadiness = () => ({
    readiness: {
      dataset_id: 'DS-J-0001', domain: 'customer', row_count: 9996, field_count: 68,
      mapped_field_count: 9, unmapped_field_count: 59, empty_field_count: 8,
      unmapped_canonical_fields: [],
      checks: [{ name: 'Match fields configured', passed: true, detail: '2 match field(s).' }],
      warnings: [], is_ready: true,
    },
    dedupe_handoff_configuration: {
      dataset_id: 'DS-J-0001', domain: 'customer', identifier_fields: ['customer_id'],
      match_fields: ['customer_name'], blocking_fields: ['city'], ignored_fields: [],
    },
  });
  const pair = (id) => ({
    pair_id: id, strategy: 'MODEL_UNCERTAIN', model_score: 0.51,
    values_a: { customer_name: 'Acme Ltd', email: 'a@x.com' },
    values_b: { customer_name: 'Acme Limited', email: 'a@x.com' },
    display_a: { customer_name: 'Acme Ltd', email: 'a@x.com' },
    display_b: { customer_name: 'Acme Limited', email: 'a@x.com' },
    evidence: { identifier_agreement: { customer_id: 'SAME' } },
    label: null,
  });
  const summary = { MATCH: 1, DISTINCT: 1, UNSURE: 1, usable_for_training: 2, note: 'n/a' };
  const model = (status, id = 'TR-2026-0001') => ({
    training_run_id: id, lifecycle_status: status, is_active: status === 'ACTIVE',
    match_fields: ['customer_name'], threshold: 0.5, threshold_calibration: 'AUTO',
    compatibility_problems: [],
  });
  const api = {
    readiness: vi.fn(async () => readyReadiness()),
    getDataset: vi.fn(async () => dataset('READY_FOR_DEDUPE')),
    listDatasets: vi.fn(async () => ({ datasets: [dataset('READY_FOR_DEDUPE')] })),
    generateCandidates: vi.fn(async () => ({ created: 1, strategy_used: 'MODEL_ASSISTED', pairs: [], instruction: 'ok' })),
    listLabelPairs: vi.fn(async () => ({ pairs: [pair('P-1')], count: 1 })),
    labelPair: vi.fn(async () => ({})),
    labelsSummary: vi.fn(async () => summary),
    runTraining: vi.fn(async () => ({
      status: 'TRAINED', training_run_id: 'TR-2026-0001', threshold: 0.5,
      threshold_calibration: 'AUTO', review_threshold: 0.7,
      label_counts: { real_steward: { match: 4, distinct: 3 }, synthetic: { used: false, mode: 'AUTO' } },
      evaluation_metrics: { pair_count: 14, precision: 0.9, recall: 0.8, f1: 0.85, true_positives: 4, false_positives: 0, false_negatives: 1, true_negatives: 9 },
      real_evaluation: { validation: null, test: { pair_count: 0 } },
    })),
    listModels: vi.fn(async () => ({ models: [model('DRAFT')] })),
    validateModel: vi.fn(async () => model('VALIDATED')),
    activateModel: vi.fn(async () => model('ACTIVE')),
    activeModel: vi.fn(async () => model('ACTIVE')),
    runMatch: vi.fn(async () => ({
      status: 'COMPLETED', match_run_id: 'MR-1', total_records: 100, candidate_pair_count: 400,
      match_count: 12, uncertain_count: 3, non_match_count: 385, threshold: 0.5,
      blocking_evaluation: { candidate_reduction_pct: 99.9, total_possible_pairs: 4950, candidate_reduction_meaning: 'blocking', candidate_recall: null, candidate_recall_note: 'no labels' },
    })),
    runCluster: vi.fn(async () => ({ status: 'COMPLETED', cluster_run_id: 'CR-1', cluster_count: 88, singleton_count: 80, largest_cluster_size: 4 })),
    generateGoldenRecords: vi.fn(async () => ({ entities_created: 88 })),
    duplicateGroups: vi.fn(async (_d, _c, offset = 0) => ({
      total_groups: 30, offset, limit: 25, columns: ['customer_name', 'email'], standardized_columns: ['customer_name', 'email'],
      groups: Array.from({ length: offset === 0 ? 25 : 5 }, (_, i) => ({
        group_id: `G${String(offset + i + 1).padStart(5, '0')}`, size: 2, confidence: 0.9,
        records: [{ row_number: 1 + i, customer_name: ' acme ltd ', email: 'A@X.COM', _std: { customer_name: 'Acme Ltd', email: 'a@x.com' } },
                  { row_number: 100 + i, customer_name: 'acme limited', email: 'A@X.COM', _std: { customer_name: 'ACME Limited', email: 'a@x.com' } }],
      })),
    })),
    submitIncomingRecords: vi.fn(async () => ({ batch: { status: 'COMPLETED', record_count: 1, attached_to_existing_count: 1, new_entity_count: 0, ambiguous_count: 0, candidate_pair_count: 3, processing_seconds: 0.1 } })),
    listIncomingBatches: vi.fn(async () => ({ batches: [] })),
    getEntities: vi.fn(async () => ({ entities: [], total: 0 })),
    listDomains: vi.fn(async () => ({ domains: ['customer', 'supplier'] })),
    domainSchema: vi.fn(async (domain) => ({ domain, fields: schemaFor(domain) })),
    pendingReviews: vi.fn(async () => ({ pending_reviews: [], count: 0, total: 0, has_more: false })),
    entitiesNeedingReview: vi.fn(async () => ({ entities: [], count: 0, total: 0, has_more: false })),
    resolvePendingReview: vi.fn(async () => ({})),
    masterData: vi.fn(async () => ({ records: [], count: 0, total_entities: 0, has_more: false })),
    trainingLabelReadiness: vi.fn(async () => ({
      domain: 'customer', match_fields: ['customer_name'],
      repository: { labelled: 9, match: 4, distinct: 3, unsure_excluded: 2, awaiting_label: 1 },
      usable: { total: 7, match: 4, distinct: 3, origin: { steward_review: 5, labeling_screen: 2 } },
      excluded_missing_match_fields: [], excluded_missing_match_fields_count: 0,
      split_preview: { train: { match: 3, distinct: 2 }, validation: { match: 1, distinct: 1 }, test: { match: 0, distinct: 0 } },
      verdict: { real_labels_sufficient: false, real_only_training_possible: false, threshold_can_be_calibrated: false },
      notes: ['Fewer than 10 real labels per class in the training split, so AUTO mode will ADD synthetic pairs next to your real ones.'],
    })),
    labelledPairs: vi.fn(async () => ({ pairs: [], count: 0, total: 0, has_more: false })),
  };
  return { api };
});

import { api } from '../src/api.js';
import ReadinessScreen from '../src/screens/ReadinessScreen.jsx';
import LabelingScreen from '../src/screens/LabelingScreen.jsx';
import TrainingScreen from '../src/screens/TrainingScreen.jsx';
import MatchingScreen from '../src/screens/MatchingScreen.jsx';
import IncomingDataScreen from '../src/screens/IncomingDataScreen.jsx';
import StewardReviewScreen from '../src/screens/StewardReviewScreen.jsx';
import MasterDataScreen from '../src/screens/MasterDataScreen.jsx';
import App from '../src/App.jsx';

const ds = () => ({ dataset_id: 'DS-J-0001', domain: 'customer', row_count: 9996, status: 'READY_FOR_DEDUPE' });

beforeEach(() => { vi.clearAllMocks(); });
afterEach(() => cleanup());

describe('Journey A-C: Readiness', () => {
  it('A: reports readiness and refreshes parent state via onReady', async () => {
    const onReady = vi.fn();
    render(<ReadinessScreen dataset={ds()} onReady={onReady} />);
    await waitFor(() => expect(screen.getByText(/✓ Ready for Dedupe/i)).toBeTruthy());
    expect(onReady).toHaveBeenCalled();
  });

  it('B: Continue CTA navigates to labeling', async () => {
    const onContinue = vi.fn();
    render(<ReadinessScreen dataset={ds()} onContinue={onContinue} />);
    await waitFor(() => expect(screen.getByTestId('readiness-continue')).toBeTruthy());
    fireEvent.click(screen.getByTestId('readiness-continue'));
    expect(onContinue).toHaveBeenCalled();
  });

  it('C: not-ready state offers Return to ER Configuration, no continue', async () => {
    api.readiness.mockResolvedValueOnce({
      readiness: { dataset_id: 'DS-J-0001', domain: 'customer', row_count: 10, field_count: 5, mapped_field_count: 1, unmapped_field_count: 4, empty_field_count: 0, unmapped_canonical_fields: [], checks: [{ name: 'X', passed: false, detail: 'no' }], warnings: [], is_ready: false },
      dedupe_handoff_configuration: { identifier_fields: [], match_fields: ['customer_name'], blocking_fields: [], ignored_fields: [] },
    });
    const onBack = vi.fn();
    render(<ReadinessScreen dataset={ds()} onBackToConfigure={onBack} />);
    await waitFor(() => expect(screen.getByRole('heading', { name: /Not ready yet/i })).toBeTruthy());
    expect(screen.queryByTestId('readiness-continue')).toBeNull();
    fireEvent.click(screen.getByTestId('readiness-back'));
    expect(onBack).toHaveBeenCalled();
  });
});

describe('Journey D-E: Labeling', () => {
  it('D: generate -> three label choices -> persist immediately -> summary refresh', async () => {
    const onContinue = vi.fn();
    render(<LabelingScreen dataset={ds()} onContinueToTraining={onContinue} />);
    await waitFor(() => expect(screen.getByText(/MATCH/)).toBeTruthy());
    // label definitions are explained
    expect(screen.getByText(/both records represent/i)).toBeTruthy();
    fireEvent.change(screen.getByPlaceholderText('e.g. priya.s'), { target: { value: 'tester' } });
    fireEvent.click(screen.getByText('Generate candidates'));
    await waitFor(() => expect(api.generateCandidates).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText('Same entity (MATCH)')).toBeTruthy());
    fireEvent.click(screen.getByText('Same entity (MATCH)'));
    await waitFor(() => expect(api.labelPair).toHaveBeenCalledWith('P-1', 'MATCH', 'tester'));
  });

  it('E: Continue-to-Training requires usable labels', async () => {
    api.labelsSummary.mockResolvedValue({ MATCH: 1, DISTINCT: 1, UNSURE: 1, usable_for_training: 2, note: 'n/a' });
    const { rerender } = render(<LabelingScreen dataset={ds()} onContinueToTraining={() => {}} />);
    const btn = await waitFor(() => screen.getByTestId('continue-to-training'));
    await waitFor(() => expect(btn.disabled).toBe(false));
    api.labelsSummary.mockResolvedValue({ MATCH: 0, DISTINCT: 0, UNSURE: 0, usable_for_training: 0, note: 'n/a' });
    rerender(<LabelingScreen dataset={{ ...ds(), dataset_id: 'DS-J-0002' }} onContinueToTraining={() => {}} />);
    const btn2 = await waitFor(() => screen.getByTestId('continue-to-training'));
    await waitFor(() => expect(btn2.disabled).toBe(true));
  });
});

describe('Journey F-H: Training lifecycle', () => {
  it('F: training completes but does NOT auto-navigate to Matching', async () => {
    const onContinueToMatching = vi.fn();
    render(<TrainingScreen dataset={ds()} onContinueToMatching={onContinueToMatching} />);
    fireEvent.click(await waitFor(() => screen.getByText('Train model')));
    await waitFor(() => expect(screen.getByText(/TRAINED/)).toBeTruthy());
    // the critical regression check: no automatic navigation
    expect(onContinueToMatching).not.toHaveBeenCalled();
  });

  it('G: DRAFT -> Validate -> VALIDATED -> Activate -> ACTIVE', async () => {
    render(<TrainingScreen dataset={ds()} />);
    await waitFor(() => screen.getByTestId('model-card-TR-2026-0001'));
    expect(screen.getByTestId('model-lifecycle').textContent).toBe('DRAFT');
    // backend state after validation, picked up by the registry refresh
    api.listModels.mockResolvedValue({ models: [{ lifecycle_status: 'VALIDATED', training_run_id: 'TR-2026-0001', is_active: false, match_fields: ['customer_name'], threshold: 0.5, threshold_calibration: 'AUTO', compatibility_problems: [] }] });
    fireEvent.click(screen.getByTestId('validate-model'));
    await waitFor(() => expect(api.validateModel).toHaveBeenCalledWith('TR-2026-0001'));
    await waitFor(() => expect(screen.getByTestId('activate-model')).toBeTruthy());
    fireEvent.click(screen.getByTestId('activate-model'));
    await waitFor(() => expect(api.activateModel).toHaveBeenCalledWith('TR-2026-0001'));
  });

  it('H: Continue-to-Matching appears only when a model is ACTIVE', async () => {
    const onContinue = vi.fn();
    api.listModels.mockResolvedValue({ models: [{ lifecycle_status: 'ACTIVE', training_run_id: 'TR-2026-0001', is_active: true, match_fields: ['customer_name'], threshold: 0.5, threshold_calibration: 'AUTO', compatibility_problems: [] }] });
    const { unmount } = render(<TrainingScreen dataset={ds()} onContinueToMatching={onContinue} />);
    await waitFor(() => expect(screen.getByTestId('continue-to-matching')).toBeTruthy());
    unmount();
    // now with only a DRAFT model, no CTA
    api.listModels.mockResolvedValue({ models: [{ lifecycle_status: 'DRAFT', training_run_id: 'TR-2', is_active: false, match_fields: ['customer_name'], threshold: 0.5, threshold_calibration: 'AUTO', compatibility_problems: [] }] });
    render(<TrainingScreen dataset={ds()} onContinueToMatching={onContinue} />);
    await waitFor(() => expect(screen.getByTestId('model-card-TR-2')).toBeTruthy());
    expect(screen.queryByTestId('continue-to-matching')).toBeNull();
    expect(onContinue).not.toHaveBeenCalled();
  });
});

describe('Journey I-K: Matching', () => {
  it('I: friendly gate (no raw 422) when no ACTIVE model', async () => {
    api.activeModel.mockRejectedValueOnce(new Error('404 No ACTIVE model'));
    render(<MatchingScreen dataset={ds()} />);
    await waitFor(() => expect(screen.getByText(/No ACTIVE model/i)).toBeTruthy());
    expect(screen.queryByText('Match')).toBeNull();
  });

  it('J: one Match click runs matching then clustering automatically and shows the actual clustered records', async () => {
    render(<MatchingScreen dataset={ds()} />);
    await waitFor(() => expect(screen.getByText(/TR-2026-0001/)).toBeTruthy());
    expect(screen.queryByText('Run matching')).toBeNull();
    expect(screen.queryByText('Run clustering')).toBeNull();
    expect(screen.queryByText('Generate Golden Records')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Match' }));
    await waitFor(() => expect(api.runCluster).toHaveBeenCalledTimes(1));
    expect(api.runMatch).toHaveBeenCalledTimes(1);
    expect(api.runCluster.mock.calls[0][1]).toBe('MR-1');           // clustered the run that was just matched
    await waitFor(() => expect(screen.getByTestId('cluster-groups')).toBeTruthy());
    // summary + first page of groups with real source values; first group expanded by default
    expect(screen.getByText(/Match & Cluster completed/)).toBeTruthy();
    expect(screen.getAllByText('ACME Limited').length).toBeGreaterThan(0);
    expect(api.duplicateGroups).toHaveBeenCalledWith('DS-J-0001', 'CR-1', 0, 25);
    // golden records are NOT generated automatically
    expect(api.generateGoldenRecords).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText('Generate Golden Records'));
    await waitFor(() => expect(api.generateGoldenRecords).toHaveBeenCalledWith('DS-J-0001', 'CR-1'));
    await waitFor(() => expect(screen.getByText(/Golden Records Generated/)).toBeTruthy());
  });

  it('J2: cluster groups are fetched a bounded page at a time and can be expanded/collapsed', async () => {
    render(<MatchingScreen dataset={ds()} />);
    fireEvent.click(await waitFor(() => screen.getByRole('button', { name: 'Match' })));
    await waitFor(() => expect(screen.getByTestId('cluster-header-G00002')).toBeTruthy());
    expect(api.duplicateGroups).toHaveBeenCalledTimes(1);           // 25 of 30 groups, not everything
    expect(screen.queryByText('Cluster G00026')).toBeNull();
    fireEvent.click(screen.getByTestId('cluster-header-G00002'));   // expand a collapsed group
    expect(screen.getAllByText('a@x.com').length).toBeGreaterThan(2);
    fireEvent.click(screen.getByText(/Show more/));
    await waitFor(() => expect(api.duplicateGroups).toHaveBeenLastCalledWith('DS-J-0001', 'CR-1', 25, 25));
    await waitFor(() => expect(screen.getByText(/Cluster G00030/)).toBeTruthy());
  });

  it('J2b: cluster records default to the STANDARDIZED values and can be switched to RAW source values', async () => {
    render(<MatchingScreen dataset={ds()} />);
    fireEvent.click(await waitFor(() => screen.getByRole('button', { name: 'Match' })));
    await waitFor(() => expect(screen.getByTestId('cluster-groups')).toBeTruthy());
    expect(screen.getAllByText('ACME Limited').length).toBeGreaterThan(0);        // standardized
    expect(screen.queryByText('acme limited')).toBeNull();
    fireEvent.change(screen.getByTestId('cluster-view-select'), { target: { value: 'raw' } });
    expect(screen.getAllByText('acme limited').length).toBeGreaterThan(0);        // raw upload value
    expect(screen.getAllByText(/match score/).length).toBeGreaterThan(0);
  });

  it('J3: matching failure -> shows the error and does NOT attempt clustering', async () => {
    api.runMatch.mockResolvedValueOnce({ status: 'FAILED', match_run_id: 'MR-X', error: 'boom in blocking' });
    render(<MatchingScreen dataset={ds()} />);
    fireEvent.click(await waitFor(() => screen.getByRole('button', { name: 'Match' })));
    await waitFor(() => expect(screen.getByText('Matching failed')).toBeTruthy());
    expect(screen.getByText(/boom in blocking/)).toBeTruthy();
    expect(api.runCluster).not.toHaveBeenCalled();
    expect(screen.queryByText('Generate Golden Records')).toBeNull();
  });

  it('J4: clustering failure -> reports matching completed, clustering failed, no golden button', async () => {
    api.runCluster.mockResolvedValueOnce({ status: 'FAILED', cluster_run_id: 'CR-X', error: 'cluster exploded' });
    render(<MatchingScreen dataset={ds()} />);
    fireEvent.click(await waitFor(() => screen.getByRole('button', { name: 'Match' })));
    await waitFor(() => expect(screen.getByText(/Clustering failed/)).toBeTruthy());
    expect(screen.getByText(/cluster exploded/)).toBeTruthy();
    expect(screen.queryByText('Generate Golden Records')).toBeNull();
    expect(screen.queryByTestId('cluster-groups')).toBeNull();
  });

  it('J5: golden record failure is reported, never shown as success', async () => {
    api.generateGoldenRecords.mockRejectedValueOnce(new Error('db locked'));
    render(<MatchingScreen dataset={ds()} />);
    fireEvent.click(await waitFor(() => screen.getByRole('button', { name: 'Match' })));
    fireEvent.click(await waitFor(() => screen.getByText('Generate Golden Records')));
    await waitFor(() => expect(screen.getByText('Golden record generation failed')).toBeTruthy());
    expect(screen.getByText(/db locked/)).toBeTruthy();
    expect(screen.queryByText(/Golden Records Generated/)).toBeNull();
  });

  it('K: after golden records, navigation to Entities/Steward Review is explicit', async () => {
    const onEntities = vi.fn();
    const onSteward = vi.fn();
    render(<MatchingScreen dataset={ds()} onGoldenRecords={onEntities} onStewardReview={onSteward} />);
    fireEvent.click(await waitFor(() => screen.getByRole('button', { name: 'Match' })));
    fireEvent.click(await waitFor(() => screen.getByText('Generate Golden Records')));
    const entBtn = await waitFor(() => screen.getByTestId('open-entities'));
    const stewBtn = screen.getByTestId('open-steward-review');
    expect(onEntities).not.toHaveBeenCalled();
    fireEvent.click(entBtn);
    fireEvent.click(stewBtn);
    expect(onEntities).toHaveBeenCalled();
    expect(onSteward).toHaveBeenCalled();
  });
});

describe('Journey L-M: Incoming data', () => {
  it('L: blocked with clear message when no ACTIVE model', async () => {
    api.activeModel.mockRejectedValueOnce(new Error('404'));
    render(<IncomingDataScreen domain="customer" />);
    await waitFor(() => expect(screen.getByText(/No ACTIVE model/i)).toBeTruthy());
    const submit = screen.getByText('Submit').closest('button');
    expect(submit.disabled).toBe(true);
  });

  it('M: shows the ACTIVE model and enables submission', async () => {
    api.activeModel.mockResolvedValue({ training_run_id: 'TR-2026-0001', lifecycle_status: 'ACTIVE', is_active: true });
    render(<IncomingDataScreen domain="customer" />);
    await waitFor(() => expect(screen.getByText(/TR-2026-0001/)).toBeTruthy());
    const submit = screen.getByText('Submit').closest('button');
    expect(submit.disabled).toBe(false);
  });
});


describe('Journey P-S: navigation, master data, steward review, incoming form', () => {
  it('P: sidebar has no Monitoring or Quick-run entries', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText('Final Master Data')).toBeTruthy());
    expect(screen.queryByText(/Monitoring/i)).toBeNull();
    expect(screen.queryByText(/Quick run/i)).toBeNull();
    expect(screen.queryByText(/Find duplicates/i)).toBeNull();
    // the remaining domain views are still there
    for (const label of ['Entities & Golden Records', 'Steward Review', 'Incoming Data', 'Survivorship Rules']) {
      expect(screen.getByText(label)).toBeTruthy();
    }
  });

  it('Q: Final Master Data shows distinct records and loads more pages', async () => {
    const rec = (i, basis, members) => ({
      golden_record_id: `GR-E${i}`, entity_id: `E${i}`, version: 1, status: 'PROPOSED', approval_basis: basis,
      member_count: members, values: { customer_name: `Name ${i}` },
    });
    api.masterData
      .mockResolvedValueOnce({ records: [rec(1, 'SINGLE_RECORD_NO_CONFLICT', 1), rec(2, 'STEWARD_APPROVED', 2)],
        count: 3, total_entities: 4, distinct_record_count: 2, merged_record_count: 1, awaiting_review_count: 1, has_more: true })
      .mockResolvedValueOnce({ records: [rec(3, 'SINGLE_RECORD_NO_CONFLICT', 1)],
        count: 3, total_entities: 4, distinct_record_count: 2, merged_record_count: 1, awaiting_review_count: 1, has_more: false });
    render(<MasterDataScreen domain="customer" />);
    await waitFor(() => expect(screen.getByText('GR-E1')).toBeTruthy());
    expect(screen.getAllByText('Distinct record').length).toBe(1);
    expect(screen.getAllByText('Approved').length).toBe(1);
    fireEvent.click(screen.getByText(/Load 50 more/));
    await waitFor(() => expect(screen.getByText('GR-E3')).toBeTruthy());
    expect(api.masterData).toHaveBeenLastCalledWith('customer', 2, 50, { scope: 'final', kind: 'all', q: '' });
    expect(screen.queryByText(/Load 50 more/)).toBeNull();
  });

  it('R: Steward Review shows the first page quickly and loads more on demand', async () => {
    const pend = (i) => ({ pending_review_id: `PR-${i}`, candidate_dataset_id: 'DS-1', candidate_record_id: String(i),
      candidate_values: { customer_name: `Cand ${i}` }, candidate_entity: null, candidate_other_values: { customer_name: `Other ${i}` },
      score: 0.7, match_threshold: 0.85, review_threshold: 0.6, model_version: 'TR-1' });
    api.pendingReviews
      .mockResolvedValueOnce({ pending_reviews: [pend(1), pend(2)], count: 2, total: 3, has_more: true })
      .mockResolvedValueOnce({ pending_reviews: [pend(3)], count: 1, total: 3, has_more: false });
    render(<StewardReviewScreen domain="customer" />);
    await waitFor(() => expect(screen.getByText('DS-1 · record 1')).toBeTruthy());
    expect(screen.getByText(/showing/i).textContent).toMatch(/2.*of.*3/);
    expect(screen.queryByText('DS-1 · record 3')).toBeNull();           // not fetched yet
    expect(api.pendingReviews).toHaveBeenCalledWith('customer', 0, 25);
    fireEvent.click(screen.getByText(/Load 25 more/));
    await waitFor(() => expect(screen.getByText('DS-1 · record 3')).toBeTruthy());
    expect(api.pendingReviews).toHaveBeenLastCalledWith('customer', 2, 25);
  });

  it('R2: resolving an item removes it and lowers the total', async () => {
    api.pendingReviews.mockResolvedValueOnce({ pending_reviews: [{
      pending_review_id: 'PR-9', candidate_dataset_id: 'DS-1', candidate_record_id: '9',
      candidate_values: { customer_name: 'X' }, candidate_entity: null, candidate_other_values: { customer_name: 'Y' },
      score: 0.7, match_threshold: 0.85, review_threshold: 0.6, model_version: 'TR-1' }], count: 1, total: 1, has_more: false });
    render(<StewardReviewScreen domain="customer" />);
    await waitFor(() => expect(screen.getByText('DS-1 · record 9')).toBeTruthy());
    fireEvent.change(screen.getByPlaceholderText('e.g. priya.s'), { target: { value: 'priya.s' } });
    fireEvent.click(screen.getByText('Reject (distinct)'));
    await waitFor(() => expect(screen.queryByText('DS-1 · record 9')).toBeNull());
    expect(api.resolvePendingReview).toHaveBeenCalledWith('PR-9', 'REJECT_MATCH', 'priya.s');
    expect(screen.getByText(/Match review.*\(0\)/)).toBeTruthy();
  });

  it('S: single-record form shows every canonical field of the selected domain', async () => {
    const { rerender } = render(<IncomingDataScreen domain="customer" />);
    for (const f of CUSTOMER_FIELDS) {
      await waitFor(() => expect(document.querySelector(`input[name="${f}"]`)).toBeTruthy());
    }
    expect(document.querySelectorAll('input[type="text"], input[type="email"], input[type="date"], input[type="tel"], input[type="url"]').length)
      .toBeGreaterThanOrEqual(CUSTOMER_FIELDS.length);
    expect(document.querySelector('input[name="dob"]').type).toBe('date');
    expect(document.querySelector('input[name="email"]').type).toBe('email');

    rerender(<IncomingDataScreen domain="supplier" />);
    await waitFor(() => expect(document.querySelector('input[name="supplier_name"]')).toBeTruthy());
    expect(document.querySelector('input[name="customer_name"]')).toBeNull();
    expect(document.querySelector('input[name="trade_name"]')).toBeTruthy();
  });

  it('S2: submitting sends canonical field names and drops empty fields', async () => {
    render(<IncomingDataScreen domain="customer" />);
    await waitFor(() => expect(document.querySelector('input[name="customer_name"]')).toBeTruthy());
    fireEvent.change(document.querySelector('input[name="customer_name"]'), { target: { value: ' Acme Ltd ' } });
    fireEvent.change(document.querySelector('input[name="duns"]'), { target: { value: '123456789' } });
    await waitFor(() => expect(screen.getByText('Submit').closest('button').disabled).toBe(false));
    fireEvent.click(screen.getByText('Submit'));
    await waitFor(() => expect(api.submitIncomingRecords).toHaveBeenCalled());
    expect(api.submitIncomingRecords).toHaveBeenCalledWith('customer', [{ customer_name: 'Acme Ltd', duns: '123456789' }]);
  });
});


describe('Journey T: reprofiling shows only FIXED, ordered record fields, labels for training', () => {
  it('T1: standardization audit lists only FIXED signals', async () => {
    api.qualityComparison = vi.fn(async () => ({ signal_audit_report: [
      { field: 'email', signal: 'CASE_VARIATION', status: 'FIXED', fixed_count: 10, remaining_count: 0, explanation: 'x' },
      { field: 'phone', signal: 'MISSING_TOKEN_VARIANTS', status: 'PARTIALLY_FIXED', fixed_count: 5, remaining_count: 2, explanation: 'y' },
      { field: 'city', signal: 'LENGTH_OUTLIERS', status: 'PRESERVED', fixed_count: 0, remaining_count: 3, explanation: 'z' },
      { field: 'dob', signal: 'DATE_FORMAT_AMBIGUOUS', status: 'NOT_APPLICABLE', fixed_count: 0, remaining_count: 0, explanation: 'w' },
    ] }));
    const { default: QualityScreen } = await import('../src/screens/QualityScreen.jsx');
    render(<QualityScreen dataset={ds()} onValidated={() => {}} />);
    await waitFor(() => expect(screen.getByText(/email · CASE_VARIATION/)).toBeTruthy());
    expect(screen.queryByText(/MISSING_TOKEN_VARIANTS/)).toBeNull();
    expect(screen.queryByText(/LENGTH_OUTLIERS/)).toBeNull();
    expect(screen.queryByText(/DATE_FORMAT_AMBIGUOUS/)).toBeNull();
    expect(screen.queryByText(/PARTIALLY FIXED/)).toBeNull();
    expect(screen.queryByText(/NOT APPLICABLE/)).toBeNull();
  });

  it('T2: Master Data card shows name + key fields and all fields in schema order', async () => {
    api.masterData.mockResolvedValueOnce({ records: [{
      golden_record_id: 'GR-E1', entity_id: 'E1', version: 1, status: 'PROPOSED', approval_basis: 'SINGLE_RECORD_NO_CONFLICT',
      member_count: 1, display_name: 'Acme Trading',
      summary: [{ field: 'customer_id', label: 'Customer id', value: 'C-1' }, { field: 'customer_name', label: 'Customer name', value: 'Acme Trading' },
                { field: 'city', label: 'City', value: 'Chennai' }],
      // deliberately alphabetical, as older stored records are
      values: { address: '1 Main Rd', city: 'Chennai', customer_id: 'C-1', customer_name: 'Acme Trading' },
    }], count: 1, matching: 1, total_entities: 1, distinct_record_count: 1, merged_record_count: 0, awaiting_count: 0, has_more: false });
    render(<MasterDataScreen domain="customer" />);
    await waitFor(() => expect(screen.getByText('GR-E1')).toBeTruthy());
    expect(screen.getByText('Acme Trading', { selector: 'div' })).toBeTruthy();     // headline name
    expect(screen.getByText('C-1')).toBeTruthy();                                    // id chip
    fireEvent.click(screen.getByText('Show all fields'));
    await waitFor(() => expect(document.querySelectorAll('.evidence-drawer .kv-row').length).toBe(4));
    const labels = [...document.querySelectorAll('.evidence-drawer .kv-row .k')].map(n => n.textContent);
    expect(labels).toEqual(['Customer id:', 'Customer name:', 'Address:', 'City:']);
  });

  it('T3: Awaiting-approval tab asks the API for scope=awaiting and shows the reason', async () => {
    api.masterData
      .mockResolvedValueOnce({ records: [], count: 0, matching: 0, total_entities: 1, awaiting_count: 1, has_more: false })
      .mockResolvedValueOnce({ records: [{
        golden_record_id: 'GR-E9', entity_id: 'E9', version: 2, status: 'PROPOSED', member_count: 2, awaiting_reason: 'RE_APPROVAL',
        awaiting_detail: 'changed', values: { customer_name: 'Beta' }, summary: [],
      }], count: 0, matching: 1, total_entities: 1, awaiting_count: 1, has_more: false });
    const onOpen = vi.fn();
    render(<MasterDataScreen domain="customer" onOpenEntity={onOpen} />);
    await waitFor(() => expect(screen.getByRole('tab', { name: /Awaiting approval/ })).toBeTruthy());
    fireEvent.click(screen.getByRole('tab', { name: /Awaiting approval/ }));
    await waitFor(() => expect(screen.getByText('GR-E9')).toBeTruthy());
    expect(api.masterData).toHaveBeenLastCalledWith('customer', 0, 50, { scope: 'awaiting', kind: 'all', q: '' });
    expect(screen.getByText('Changed since approval')).toBeTruthy();
    fireEvent.click(screen.getByText(/Review & approve/));
    expect(onOpen).toHaveBeenCalledWith('E9');
  });

  it('T4: Steward Review tells the steward what Distinct did and shows fields in order', async () => {
    api.pendingReviews.mockResolvedValueOnce({ pending_reviews: [{
      pending_review_id: 'PR-5', candidate_dataset_id: 'DS-1', candidate_record_id: '5',
      candidate_values: { address: 'Road 1', customer_name: 'Gamma', customer_id: 'C-5' },
      candidate_entity: { entity_id: 'ENT-1', golden_values: { city: 'Pune', customer_name: 'Gamma Corp' } }, candidate_other_values: null,
      score: 0.7, match_threshold: 0.85, review_threshold: 0.6, model_version: 'TR-1' }], count: 1, total: 1, has_more: false });
    api.resolvePendingReview.mockResolvedValueOnce({ decision: 'REJECT_MATCH', will_appear_in_final_master_data: true,
      counterpart_will_appear_in_final_master_data: false, mirror_items_resolved: 0 });
    render(<StewardReviewScreen domain="customer" />);
    await waitFor(() => expect(screen.getByText('DS-1 · record 5')).toBeTruthy());
    // schema order: Customer id, Customer name, Address (not alphabetical Address first)
    await waitFor(() => {
      const ks = [...document.querySelectorAll('.evidence-drawer .kv-row .k')].map(n => n.textContent);
      expect(ks.slice(0, 3)).toEqual(['Customer id:', 'Customer name:', 'Address:']);
    });
    fireEvent.change(screen.getByPlaceholderText('e.g. priya.s'), { target: { value: 'priya.s' } });
    fireEvent.click(screen.getByText('Reject (distinct)'));
    await waitFor(() => expect(screen.getByText(/Marked DISTINCT/)).toBeTruthy());
    expect(screen.getByText(/still awaiting approval/)).toBeTruthy();
  });

  it('T5: Training screen previews the label repository and what AUTO will do', async () => {
    render(<TrainingScreen dataset={ds()} />);
    await waitFor(() => expect(screen.getByTestId('label-repository-panel')).toBeTruthy());
    const panel = screen.getByTestId('label-repository-panel');
    expect(panel.textContent).toMatch(/usable by this model\s*7/);
    expect(panel.textContent).toMatch(/from Steward Review\s*5/);
    expect(panel.textContent).toMatch(/synthetic pairs are added next to them/);
    expect(api.trainingLabelReadiness).toHaveBeenCalledWith('DS-J-0001');
  });
});
