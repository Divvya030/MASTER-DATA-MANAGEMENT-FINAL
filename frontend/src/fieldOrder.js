import { useEffect, useState } from 'react';
import { api } from './api';

// One schema fetch per domain per page load.
const cache = new Map();

export function loadDomainSchema(domain) {
  if (!cache.has(domain)) {
    cache.set(domain, Promise.resolve().then(() => api.domainSchema(domain))
      .catch(e => { cache.delete(domain); throw e; }));
  }
  return cache.get(domain);
}

// {fields:[{name,label,...}]} for the domain, or null until it has loaded.
export function useDomainSchema(domain) {
  const [schema, setSchema] = useState(null);
  useEffect(() => {
    let live = true;
    setSchema(null);
    if (domain) loadDomainSchema(domain).then(s => { if (live) setSchema(s); }).catch(() => {});
    return () => { live = false; };
  }, [domain]);
  return schema;
}

export function prettyLabel(name) {
  const t = String(name).replace(/_/g, ' ');
  return t.charAt(0).toUpperCase() + t.slice(1);
}

// Entries of a {field: value} object in schema order (id, name, legal name, ... address ...);
// fields the schema does not know come last, alphabetically. Without a schema the object's own
// order is kept (the backend already sends it in schema order).
export function orderedEntries(values, schema) {
  const entries = Object.entries(values || {});
  if (!schema?.fields) return entries;
  const pos = new Map(schema.fields.map((f, i) => [f.name, i]));
  return entries.sort(([a], [b]) => {
    const pa = pos.has(a) ? pos.get(a) : 1e6;
    const pb = pos.has(b) ? pos.get(b) : 1e6;
    return pa - pb || a.localeCompare(b);
  });
}

export function labelFor(name, schema) {
  return schema?.fields?.find(f => f.name === name)?.label || prettyLabel(name);
}

export function isBlank(v) {
  return v == null || String(v).trim() === '';
}
