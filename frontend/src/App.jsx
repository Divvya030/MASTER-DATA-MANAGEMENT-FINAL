import React, { useEffect, useRef, useState, useCallback } from 'react';
import './theme.css';
import { api } from './api';
import UploadScreen from './screens/UploadScreen';
import MappingScreen from './screens/MappingScreen';
import ProfilingScreen from './screens/ProfilingScreen';
import StandardizeScreen from './screens/StandardizeScreen';
import QualityScreen from './screens/QualityScreen';
import ERConfigScreen from './screens/ERConfigScreen';
import ReadinessScreen from './screens/ReadinessScreen';
import LabelingScreen from './screens/LabelingScreen';
import TrainingScreen from './screens/TrainingScreen';
import MatchingScreen from './screens/MatchingScreen';
import EntitiesScreen from './screens/EntitiesScreen';
import StewardReviewScreen from './screens/StewardReviewScreen';
import IncomingDataScreen from './screens/IncomingDataScreen';
import SurvivorshipScreen from './screens/SurvivorshipScreen';
import MasterDataScreen from './screens/MasterDataScreen';

const STAGES = [
  { key: 'ingest', label: 'Ingest' },
  { key: 'map', label: 'Map' },
  { key: 'profile', label: 'Profile' },
  { key: 'standardize', label: 'Cleanse' },
  { key: 'validate', label: 'Validate' },
  { key: 'configure', label: 'Configure ER' },
  { key: 'review', label: 'Readiness' },
  { key: 'label', label: 'Label (v2, optional)' },
  { key: 'train', label: 'Train (v2, optional)' },
  { key: 'match', label: 'Match & Cluster' },
];

// Domain-wide screens: not tied to any one dataset -- the mastered entity
// store, review queue and incoming data all span every dataset
// ever ingested for a domain, including deleted ones.
const DOMAIN_VIEWS = [
  { key: 'entities', label: 'Entities & Golden Records' },
  { key: 'steward', label: 'Steward Review' },
  { key: 'master', label: 'Final Master Data' },
  { key: 'incoming', label: 'Incoming Data' },
  { key: 'survivorship', label: 'Survivorship Rules' },
];

const STATUS_TO_STAGE_INDEX = {
  UPLOADED: 0,
  MAPPED: 1,
  PROFILED: 2,
  STANDARDIZED: 3,
  VALIDATED: 4,
  ER_CONFIGURED: 5,
  READY_FOR_DEDUPE: 6,
};

export default function App() {
  const [datasets, setDatasets] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [activeDataset, setActiveDataset] = useState(null);
  const [activeStage, setActiveStage] = useState('ingest');
  // 'pipeline' = per-dataset onboarding flow above; any DOMAIN_VIEWS key =
  // a domain-wide screen (entity store, review queue, incoming data,
  // survivorship rules) that spans every dataset in that domain,
  // including deleted ones -- so it is deliberately NOT tied to activeId.
  const [view, setView] = useState('pipeline');
  const [workspaceDomain, setWorkspaceDomain] = useState('customer');
  const [error, setError] = useState(null);
  const [navOpen, setNavOpen] = useState(false);   // off-canvas rail on small screens
  const [focusEntityId, setFocusEntityId] = useState(null);   // opens Entities straight on one entity

  const refreshDatasets = useCallback(() => {
    api.listDatasets().then(r => setDatasets(r.datasets)).catch(e => setError(e.message));
  }, []);

  useEffect(() => { refreshDatasets(); }, [refreshDatasets]);

  useEffect(() => {
    if (!activeId) { setActiveDataset(null); return; }
    api.getDataset(activeId).then(ds => { setActiveDataset(ds); setWorkspaceDomain(ds.domain); }).catch(e => setError(e.message));
  }, [activeId]);

  const reachedIndex = activeDataset ? (STATUS_TO_STAGE_INDEX[activeDataset.status] ?? 0) : -1;
  // Training/labeling/matching aren't tracked by dataset.status (they're
  // separate ER runs, not ingestion stages), so once a dataset is
  // READY_FOR_DEDUPE the remaining pipeline stages become clickable too.
  const maxClickable = reachedIndex >= 6 ? STAGES.length - 1 : reachedIndex;

  function selectDataset(id) {
    setNavOpen(false);
    setError(null);
    setView('pipeline');
    setActiveId(id);
    setActiveStage('map');
  }

  function selectDomainView(key, entityId = null) {
    setNavOpen(false);
    setError(null);
    setFocusEntityId(key === 'entities' ? entityId : null);
    setView(key);
  }

  async function deleteDataset(id) {
    const confirmed = window.confirm(
      `Delete dataset ${id}? This permanently removes its raw file, mapping, profiling, and ER configuration. This cannot be undone.`
    );
    if (!confirmed) return;
    setError(null);
    try {
      await api.deleteDataset(id);
      if (id === activeId) {
        setActiveId(null);
        setActiveStage('ingest');
      }
      refreshDatasets();
    } catch (e) {
      setError(e.message);
    }
  }

  function onUploaded(ds) {
    setError(null);
    refreshDatasets();
    setView('pipeline');
    setActiveId(ds.dataset_id);
    setActiveStage('map');
  }

  function goto(stageKey) {
    setActiveStage(stageKey);
  }

  function afterDatasetMutated() {
    if (activeId) api.getDataset(activeId).then(setActiveDataset);
    refreshDatasets();
  }

  const currentStageLabel = STAGES.find(s => s.key === activeStage)?.label;

  return (
    <div className={`shell${navOpen ? ' nav-open' : ''}`}>
      {navOpen && <div className="rail-scrim" onClick={() => setNavOpen(false)} aria-hidden="true" />}
      <aside className="rail" aria-label="Navigation">
        <div className="rail-brand">Meridian<small>MDM CONTROL CENTER</small></div>

        <div>
          <div className="rail-section-label">Datasets</div>
          <div className="dataset-list">
            {datasets.map(ds => (
              <div
                key={ds.dataset_id}
                className={`dataset-item ${ds.dataset_id === activeId ? 'active' : ''}`}
                onClick={() => selectDataset(ds.dataset_id)}
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}
              >
                <div style={{ minWidth: 0 }}>
                  <div className="ds-name">{ds.dataset_id}</div>
                  <div className="ds-meta">{ds.domain} · {ds.row_count.toLocaleString()} rows · {ds.status.replace(/_/g, ' ')}</div>
                </div>
                <button
                  title={`Delete ${ds.dataset_id}`}
                  onClick={(e) => { e.stopPropagation(); deleteDataset(ds.dataset_id); }}
                  style={{
                    background: 'none', border: 'none', color: 'var(--slate-2)',
                    cursor: 'pointer', fontSize: 14, padding: '2px 4px', flexShrink: 0,
                  }}
                >
                  ✕
                </button>
              </div>
            ))}
            {datasets.length === 0 && (
              <div className="ds-meta">No datasets yet.</div>
            )}
          </div>
        </div>

        <div>
          <div className="rail-section-label">Domain workspace</div>
          <select value={workspaceDomain} onChange={e => setWorkspaceDomain(e.target.value)}
                  style={{ width: '100%', padding: 8, marginBottom: 8, borderRadius: 4, fontSize: 12.5 }}>
            <option value="customer">customer</option>
            <option value="supplier">supplier</option>
          </select>
          <div className="dataset-list">
            {DOMAIN_VIEWS.map(v => (
              <div key={v.key} className={`dataset-item ${view === v.key ? 'active' : ''}`}
                   onClick={() => selectDomainView(v.key)}>
                <div className="ds-name" style={{ fontSize: 12.5, fontFamily: 'var(--font-body)' }}>{v.label}</div>
              </div>
            ))}
          </div>
        </div>

        <button className="rail-new-btn" onClick={() => { setNavOpen(false); setView('pipeline'); setActiveId(null); setActiveStage('ingest'); }}>
          + New dataset
        </button>
      </aside>

      <div className="main-col">
        <header className="topbar">
          <div className="topbar-left">
            <button className="nav-toggle" aria-label={navOpen ? 'Close navigation' : 'Open navigation'}
                    aria-expanded={navOpen} onClick={() => setNavOpen(o => !o)}>{navOpen ? '✕' : '☰'}</button>
            <span className="topbar-brand">Meridian</span>
            {view !== 'pipeline' ? (
              <span className="topbar-crumb">Domain workspace <span className="crumb-sep">/</span> <b>{DOMAIN_VIEWS.find(v => v.key === view)?.label}</b></span>
            ) : activeDataset ? (
              <span className="topbar-crumb">
                {activeDataset.domain}
                <span className="crumb-sep">/</span>
                <b className="topbar-ds">{activeDataset.dataset_id}</b>
                <StatusChip status={activeDataset.status} />
                <span className="crumb-sep">/</span>
                <span className="topbar-stage">{currentStageLabel}</span>
              </span>
            ) : (
              <span className="topbar-crumb">New dataset</span>
            )}
          </div>
          <div className="topbar-right">
            <span className="topbar-pill">{datasets.length} dataset{datasets.length === 1 ? '' : 's'}</span>
          </div>
        </header>

        <main className="workspace">
          {error && <div className="status-banner error">{error}</div>}

          {view !== 'pipeline' ? (
            <DomainWorkspace view={view} domain={workspaceDomain} focusEntityId={focusEntityId}
                             go={selectDomainView} />
          ) : (
            <>
              {activeDataset && (
                <PipelineTrack maxClickable={maxClickable} activeStage={activeStage} onSelect={goto} />
              )}

              {activeStage === 'ingest' || !activeDataset ? (
                <UploadScreen onUploaded={onUploaded} />
              ) : activeStage === 'map' ? (
                <MappingScreen dataset={activeDataset} onMapped={() => { afterDatasetMutated(); goto('profile'); }} />
              ) : activeStage === 'profile' ? (
                <ProfilingScreen dataset={activeDataset} onProfiled={() => { afterDatasetMutated(); goto('standardize'); }} />
              ) : activeStage === 'standardize' ? (
                <StandardizeScreen dataset={activeDataset} onStandardized={() => { afterDatasetMutated(); goto('validate'); }} />
              ) : activeStage === 'validate' ? (
                <QualityScreen dataset={activeDataset} onValidated={() => { afterDatasetMutated(); goto('configure'); }} />
              ) : activeStage === 'configure' ? (
                <ERConfigScreen dataset={activeDataset} onConfigured={() => { afterDatasetMutated(); goto('review'); }} />
              ) : activeStage === 'review' ? (
                <ReadinessScreen dataset={activeDataset} onReady={afterDatasetMutated}
                                 onContinue={() => { afterDatasetMutated(); goto('label'); }}
                                 onSkipToMatch={() => { afterDatasetMutated(); goto('match'); }}
                                 onBackToConfigure={() => goto('configure')} />
              ) : activeStage === 'label' ? (
                <LabelingScreen dataset={activeDataset} onContinueToTraining={() => goto('train')} />
              ) : activeStage === 'train' ? (
                <TrainingScreen dataset={activeDataset}
                                onLabelMore={() => goto('label')}
                                onContinueToMatching={() => goto('match')}
                                datasetReady={maxClickable >= 6} />
              ) : (
                <MatchingScreen dataset={activeDataset}
                                onGoldenRecords={() => selectDomainView('entities')}
                                onStewardReview={() => selectDomainView('steward')} />
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function StatusChip({ status }) {
  const cls = {
    READY_FOR_DEDUPE: 'badge-good',
    ER_CONFIGURED: 'badge-warn',
    VALIDATED: 'badge-warn',
    FAILED: 'badge-bad',
  }[status] || 'badge-neutral';
  return <span className={`badge ${cls} topbar-status`}>{status.replace(/_/g, ' ')}</span>;
}

function DomainWorkspace({ view, domain, focusEntityId, go }) {
  switch (view) {
    case 'entities':
      // key: opening a different entity from another screen remounts the list on that entity
      return <EntitiesScreen key={focusEntityId || 'list'} domain={domain} initialEntityId={focusEntityId}
                             onOpenMasterData={() => go('master')} />;
    case 'steward':
      return <StewardReviewScreen domain={domain} onOpenMasterData={() => go('master')}
                                  onOpenEntity={(id) => go('entities', id)} />;
    case 'master': return <MasterDataScreen domain={domain} onOpenEntity={(id) => go('entities', id)} />;
    case 'incoming': return <IncomingDataScreen domain={domain} />;
    case 'survivorship': return <SurvivorshipScreen domain={domain} />;
    default: return null;
  }
}

function PipelineTrack({ maxClickable, activeStage, onSelect }) {
  const activeIdx = STAGES.findIndex(s => s.key === activeStage);
  const trackRef = useRef(null);
  // keep the current step visible when the track scrolls horizontally on narrow screens
  useEffect(() => {
    const el = trackRef.current?.querySelector('.pipeline-node.current');
    if (el && typeof el.scrollIntoView === 'function') el.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' });
  }, [activeStage]);
  return (
    <div className="pipeline-track" ref={trackRef}>
      {STAGES.map((s, i) => (
        <React.Fragment key={s.key}>
          {i > 0 && <div className={`pipeline-connector ${i <= maxClickable ? 'done' : ''}`} />}
          <div
            className={`pipeline-node ${i < maxClickable ? 'done' : ''} ${i === activeIdx ? 'current' : ''}`}
            onClick={() => i <= Math.max(maxClickable, activeIdx) && onSelect(s.key)}
          >
            <div className="pipeline-dot">{i < maxClickable ? '✓' : i + 1}</div>
            <div className="pipeline-label">{s.label}</div>
          </div>
        </React.Fragment>
      ))}
    </div>
  );
}
