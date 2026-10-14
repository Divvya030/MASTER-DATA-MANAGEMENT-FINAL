import React from 'react';
import { orderedEntries, labelFor, isBlank } from '../fieldOrder';

// A record's fields in schema order (id, name, ... address ...) with readable labels.
// Used wherever a record is shown field by field, so every screen lists them the same way.
export default function RecordFields({ values, schema, showMissing = false, compact = true }) {
  const rows = orderedEntries(values, schema).filter(([, v]) => showMissing || !isBlank(v));
  if (rows.length === 0) return <i style={{ fontSize: 12 }}>no values</i>;
  return (
    <div>
      {rows.map(([k, v]) => (
        <div key={k} className="kv-row" style={{ fontSize: compact ? 12 : 12.5, marginBottom: 3 }}>
          <span className="k" style={{ color: 'var(--slate)' }}>{labelFor(k, schema)}:</span>{' '}
          <span style={{ minWidth: 0, wordBreak: 'break-word' }}>{isBlank(v) ? <i>missing</i> : String(v)}</span>
        </div>
      ))}
    </div>
  );
}

// The few fields that identify a record on a list card: [{field,label,value}] from the backend.
export function SummaryChips({ items }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="metric-row" style={{ marginTop: 6 }}>
      {items.map(it => (
        <div className="metric" key={it.field}>{it.label} <b>{String(it.value).slice(0, 48)}</b></div>
      ))}
    </div>
  );
}
