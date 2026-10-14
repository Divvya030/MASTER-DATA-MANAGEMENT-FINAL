import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

// Mock the api module with canned responses mirroring real backend shapes
vi.mock('../src/api.js', () => {
  const dataset = {
    dataset_id: 'DS-2026-0001', original_filename: 'customer.csv', domain: 'customer',
    row_count: 9996, column_count: 68, columns: ['customer_name', 'account_number'],
    status: 'UPLOADED', header_warnings: [], sha256: 'a'.repeat(64),
  };
  return {
    api: {
      listDomains: vi.fn().resolves ? undefined : vi.fn(async () => ({ domains: ['customer', 'supplier'] })),
      listDatasets: vi.fn(async () => ({ datasets: [dataset] })),
      getDataset: vi.fn(async () => dataset),
      upload: vi.fn(async () => dataset),
      mappingSuggestions: vi.fn(async () => ({
        dataset_id: dataset.dataset_id,
        canonical_fields: ['customer_name', 'customer_id'],
        suggestions: {
          customer_name: [{ source_column: 'customer_name', confidence: 1.0, match_type: 'EXACT', reason: 'exact', completeness: 99, distinct_count: 100, sample_values: ['Acme Corp'], is_empty: false }],
          customer_id: [],
        },
        empty_ignored_columns: ['unused_col'],
      })),
      getMapping: vi.fn(async () => ({ mapping: {} })),
      confirmMapping: vi.fn(async () => ({ mapping: { customer_name: 'customer_name' } })),
      runProfile: vi.fn(async () => ({ profiles: {
        customer_name: {
          canonical_field: 'customer_name', source_column: 'customer_name', stage: 'RAW',
          data_quality: {
            completeness: { total_rows: 100, non_null_count: 99, null_count: 1, null_percentage: 1.0, missing_token_breakdown: {} },
            cardinality: { distinct_count: 90, uniqueness_ratio: 0.9, duplicate_value_count: 10, is_constant: false, top_values: [{ value: 'Acme', count: 3 }], top_value_share: 0.03 },
            type_inference: { inferred_type: 'FREE_TEXT', type_confidence: 100, type_distribution: {} },
            string_quality: { leading_trailing_whitespace_count: 0, case_variation_detected: false, repeated_internal_whitespace_count: 0, unusual_control_char_count: 0, length_min: 3, length_max: 20, length_mean: 10, length_outlier_count: 0 },
            format_patterns: { format_patterns: [], is_mixed_format: false },
            identifier_quality: null, date_ambiguity: null,
          },
          cleansing_signals: [], er_readiness: 'HIGH', er_recommendation: 'GOOD_MATCH_FIELD',
          blocking_assessment: 'GOOD_BLOCKING_CANDIDATE', reasons: ['high completeness'],
        },
      } })),
      getProfile: vi.fn(async () => { throw new Error('not found'); }),
      runStandardize: vi.fn(async () => ({ transformation_summary: { TRIM_COLLAPSE_WHITESPACE: 5 }, transformation_count: 5 })),
      runReprofile: vi.fn(async () => ({ profiles: {} })),
      qualityComparison: vi.fn(async () => ({ comparisons: {
        customer_name: {
          canonical_field: 'customer_name',
          before: { completeness_pct: 98, whitespace_issues: 5, is_mixed_format: false, scientific_notation_count: 0, cleansing_signal_count: 1, er_recommendation: 'GOOD_MATCH_FIELD', blocking_assessment: 'GOOD_BLOCKING_CANDIDATE' },
          after: { completeness_pct: 98, whitespace_issues: 0, is_mixed_format: false, scientific_notation_count: 0, cleansing_signal_count: 0, er_recommendation: 'GOOD_MATCH_FIELD', blocking_assessment: 'GOOD_BLOCKING_CANDIDATE' },
          improved: true, notes: ['Whitespace issues reduced from 5 to 0.'],
        },
      }, signal_audit_report: [
        { field: 'customer_name', signal: 'LEADING_TRAILING_WHITESPACE', detected_before: true, changed: true, fixed_count: 5, remaining_count: 0, status: 'FIXED', explanation: 'Leading and trailing whitespace was trimmed.', is_estimated: false },
      ] })),
      erConfigSuggestions: vi.fn(async () => ({ fields: {
        // Deliberately suggests BOTH roles, to exercise the Item #3 multi-role fix in the UI.
        customer_name: { canonical_field: 'customer_name', er_recommendation: 'GOOD_MATCH_FIELD', blocking_assessment: 'GOOD_BLOCKING_CANDIDATE', er_readiness: 'HIGH', suggested_roles: ['MATCH', 'BLOCKING'], completeness_pct: 98, reasons: ['High completeness and uniqueness.'] },
      } })),
      submitErConfig: vi.fn(async () => ({ configuration: { identifier_fields: [], match_fields: ['customer_name'], blocking_fields: ['customer_name'], ignored_fields: [] }, warnings: [] })),
      readiness: vi.fn(async () => ({
        readiness: {
          dataset_id: dataset.dataset_id, domain: 'customer', row_count: 9996, field_count: 68,
          mapped_field_count: 9, unmapped_field_count: 59, empty_field_count: 8,
          unmapped_canonical_fields: ['email', 'phone'],
          checks: [{ name: 'Dataset ingested', passed: true, detail: '9996 rows' },
                   { name: 'Match fields configured', passed: true, detail: '1 match field(s).' }],
          warnings: ['No strong identifier configured.'], is_ready: true,
        },
        dedupe_handoff_configuration: {
          dataset_id: dataset.dataset_id, domain: 'customer', identifier_fields: [],
          match_fields: ['customer_name'], blocking_fields: ['customer_name'], ignored_fields: ['country'],
          standardization_stage: 'STANDARDIZED',
        },
      })),
    },
  };
});

import App from '../src/App.jsx';

describe('App smoke test (all 7 module screens)', () => {
  it('renders the upload screen without a dataset selected', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText(/Ingest a source file/i)).toBeTruthy());
  });

  it('renders the dataset list and can navigate through every stage', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText('DS-2026-0001')).toBeTruthy());

    fireEvent.click(screen.getByText('DS-2026-0001'));
    await waitFor(() => expect(screen.getByText(/Map source columns/i)).toBeTruthy());

    // Follow the real flow: map a field, then confirm -- this is what actually
    // advances reachedIndex and unlocks later pipeline stages (by design).
    await waitFor(() => expect(screen.getAllByRole('button', { name: 'Map' }).length).toBeGreaterThan(0));
    fireEvent.click(screen.getAllByRole('button', { name: 'Map' })[0]);
    fireEvent.click(screen.getByText('Confirm mapping →'));
    await waitFor(() => expect(screen.getByText(/Data quality profile/i)).toBeTruthy());

    fireEvent.click(screen.getByText('Run profiling'));
    await waitFor(() => expect(screen.getByText('customer_name')).toBeTruthy());
    fireEvent.click(screen.getByText('Continue to standardization →'));

    await waitFor(() => expect(screen.getByText(/Standardize & cleanse/i)).toBeTruthy());
    fireEvent.click(screen.getByText('Run standardization'));
    await waitFor(() => expect(screen.getByText(/trim collapse whitespace/i)).toBeTruthy());
    fireEvent.click(screen.getByText('Continue to validation →'));

    await waitFor(() => expect(screen.getByText(/Standardization audit report/i)).toBeTruthy());
    fireEvent.click(screen.getByText('Continue to ER configuration →'));

    await waitFor(() => expect(screen.getByText(/Configure ER field roles/i)).toBeTruthy());
    // Item #3 regression in the actual rendered UI: a field suggested for both
    // roles must show both, not collapse to one.
    await waitFor(() => expect(screen.getByText('suggested: MATCH + BLOCKING')).toBeTruthy());
    fireEvent.click(screen.getByText('Confirm ER configuration →'));

    await waitFor(() => expect(screen.getByText(/Final pre-dedupe review/i)).toBeTruthy());
    await waitFor(() => expect(screen.getByText(/Ready for Dedupe/i)).toBeTruthy());
    // Explicit CTA must be present once ready (no reliance on clickable nodes)
    expect(screen.getByTestId('readiness-continue')).toBeTruthy();
  });
});
