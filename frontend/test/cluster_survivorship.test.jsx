import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent, cleanup, within } from '@testing-library/react';

vi.mock('../src/api.js', () => {
  const api = {
    duplicateGroups: vi.fn(async () => ({
      total_groups: 1, offset: 0, limit: 25,
      columns: ['customer_name', 'city', 'source_system', 'last_updated'],
      standardized_columns: ['customer_name', 'city', 'source_system', 'last_updated'],
      mapping: { customer_name: 'customer_name', city: 'city', source_system: 'source_system', last_updated: 'last_updated' },
      metadata_fields: ['last_updated', 'source_system'],
      source_system_field: 'source_system', recency_field: 'last_updated',
      groups: [{
        group_id: 'G00001', size: 2, confidence: 0.82, match_score: 0.82,
        records: [
          { row_number: 3, customer_name: 'Rohan Nair', city: 'Mumbai', source_system: 'Oracle CX', last_updated: '2026-01-25 09:01:54',
            _std: { customer_name: 'Rohan Nair', city: 'Mumbai', source_system: 'Oracle CX', last_updated: '2026-01-25 09:01:54' } },
          { row_number: 90, customer_name: 'Rohan Nair', city: 'Pune', source_system: 'Salesforce CRM', last_updated: '2026-08-30 17:20:00',
            _std: { customer_name: 'Rohan Nair', city: 'Pune', source_system: 'Salesforce CRM', last_updated: '2026-08-30 17:20:00' } },
        ],
      }],
    })),
    getSurvivorship: vi.fn(async () => ({
      domain: 'customer', is_default: true, lifecycle_status: null, generates_golden_records: true,
      config: { version: 1, rule_order: ['source_priority', 'recency', 'validity', 'frequency', 'completeness'],
                field_rules: {}, source_priority: {}, recency_field: 'last_updated', source_system_field: 'source_system',
                timestamp_dayfirst: true },
    })),
    detectedSurvivorship: vi.fn(async () => ({
      source_systems: [{ name: 'Oracle CX', records: 177 }, { name: 'Salesforce CRM', records: 151 }],
      datasets: [{ dataset_id: 'DS-1', source_system_mapped: true, recency_mapped: true, timestamps_parsed_pct: 100 }],
    })),
    domainSchema: vi.fn(async () => ({ fields: [
      { name: 'customer_name', label: 'Customer name' }, { name: 'last_updated', label: 'Last updated', metadata: true },
      { name: 'source_system', label: 'Source system', metadata: true }] })),
    putSurvivorship: vi.fn(async () => ({})),
    submitSurvivorship: vi.fn(), confirmSurvivorship: vi.fn(),
  };
  return { api };
});

import { api } from '../src/api.js';
import ClusterGroupsView from '../src/components/ClusterGroupsView.jsx';
import SurvivorshipScreen from '../src/screens/SurvivorshipScreen.jsx';

beforeEach(() => vi.clearAllMocks());
afterEach(() => cleanup());

describe('Clustered records comparison view', () => {
  it('puts source system + timestamp in each record header and flags the most recent record', async () => {
    render(<ClusterGroupsView datasetId="DS-1" clusterRunId="CR-1" />);
    await waitFor(() => screen.getByTestId('cluster-groups'));
    expect(screen.getAllByText('Oracle CX').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Salesforce CRM').length).toBeGreaterThan(0);
    expect(screen.getByText('2026-08-30 17:20:00')).toBeTruthy();
    expect(screen.getByText(/most recent/)).toBeTruthy();
    // metadata is a header, never a comparison row
    expect(screen.queryByRole('rowheader', { name: /source system/i })).toBeNull();
  });

  it('highlights fields that differ and can hide the ones that agree', async () => {
    render(<ClusterGroupsView datasetId="DS-1" clusterRunId="CR-1" />);
    await waitFor(() => screen.getByTestId('cluster-groups'));
    expect(screen.getByText('Mumbai').className).toContain('cell-diff');
    expect(screen.getByText('Pune').className).toContain('cell-diff');
    expect(screen.getAllByText('Rohan Nair').some(el => el.className.includes('cell-same'))).toBe(true);
    expect(screen.getByText('1 field differ')).toBeTruthy();
    fireEvent.click(screen.getByLabelText('Only fields that differ'));
    expect(screen.queryByRole('rowheader', { name: /customer name/i })).toBeNull();
    expect(screen.getByRole('rowheader', { name: /city/i })).toBeTruthy();
  });

  it('header is keyboard accessible', async () => {
    render(<ClusterGroupsView datasetId="DS-1" clusterRunId="CR-1" />);
    const head = await waitFor(() => screen.getByTestId('cluster-header-G00001'));
    expect(head.getAttribute('aria-expanded')).toBe('true');
    fireEvent.keyDown(head, { key: 'Enter' });
    expect(head.getAttribute('aria-expanded')).toBe('false');
  });
});

describe('Survivorship screen', () => {
  it('shows what the data offers and saves the recency field + source priorities', async () => {
    render(<SurvivorshipScreen domain="customer" />);
    await waitFor(() => screen.getByText('Survivorship configuration'));
    await waitFor(() => expect(screen.getByText(/Found 2 source systems/)).toBeTruthy());
    expect(screen.getAllByText('data available').length).toBe(2);        // source_priority + recency are live

    fireEvent.click(screen.getByText(/Oracle CX/, { selector: 'button' }));
    expect(screen.getByDisplayValue('Oracle CX')).toBeTruthy();
    fireEvent.click(screen.getByText('Save draft'));
    await waitFor(() => expect(api.putSurvivorship).toHaveBeenCalled());
    const [, payload] = api.putSurvivorship.mock.calls[0];
    expect(payload.recency_field).toBe('last_updated');
    expect(payload.timestamp_dayfirst).toBe(true);
    expect(payload.source_priority).toEqual({ 'Oracle CX': 1 });
  });

  it('warns when no source system is mapped', async () => {
    api.detectedSurvivorship.mockResolvedValueOnce({ source_systems: [], datasets: [{ source_system_mapped: false, recency_mapped: false }] });
    render(<SurvivorshipScreen domain="customer" />);
    await waitFor(() => expect(screen.getByText(/No dataset in this domain maps a column/)).toBeTruthy());
    expect(screen.getByText('needs a mapped source system')).toBeTruthy();
  });
});
