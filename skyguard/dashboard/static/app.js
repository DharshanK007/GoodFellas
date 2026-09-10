/* ================================================================
   SkyGuard AI — app.js v3.2 (Leaflet Canvas Engine + Zero Watermark)
   OpenStreetMap Base · 60fps Canvas Markers · Flowing DACM Vectors
   Modal Gated Ingestion · Live Telemetry · Hybrid Light/Dark Theme
   ================================================================ */

'use strict';

/* ── STATUS COLOUR MAP ─────────────────────────────────────── */
const STATUS_COLOR = {
  NORMAL:                          '#16a34a',
  STATION_SENSOR_FAULT:            '#dc2626',
  GENUINE_METEOROLOGICAL_EVENT:    '#9333ea',
  UNCERTAIN_INSUFFICIENT_EVIDENCE: '#d97706',
  UNANALYSED:                      '#64748b',
  HEALTHY:                         '#16a34a',
  DEGRADED:                        '#d97706',
  FAILED:                          '#dc2626',
};

/* ── GLOBAL STATE ──────────────────────────────────────────── */
const STATE = {
  catalog:         [],
  activeSubnet:    [],
  selectedStation: null,
  pendingStation:  null,
  pendingPreselect:null,
  currentMode:     'REAL',
  currentTimestep: 0,
  totalTimesteps:  0,
  markerResults:   {},
  stationReadings: {},
  isPlaying:       false,
  playInterval:    null,
  playSpeed:       1500,
  auditRunning:    false,
  auditResults:    null,
  map:             null,
  stationLayer:    null,
  dacmLayer:       null,
  windLayer:       null,   // wind vector indicator at selected station
  pulseLayer:      null,
  waveLayer:       null,
  canvasRenderer:  null,
  alertsInterval:  null,
  // IMD Nowcast layer
  nowcastLayer:    null,
  nowcastVisible:  true,
  // Pipeline Telemetry & Logs
  pipelineLogIndex:0,
  pipelineInterval:null,
};

/* ── API HELPERS ───────────────────────────────────────────── */
async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined
  });
  if (!r.ok) { const t = await r.text(); throw new Error(`POST ${path} → ${r.status}: ${t}`); }
  return r.json();
}
async function apiPostQuery(path, params) {
  const url = path + '?' + new URLSearchParams(params).toString();
  const r   = await fetch(url, { method: 'POST' });
  if (!r.ok) { const t = await r.text(); throw new Error(`POST ${url} → ${r.status}: ${t}`); }
  return r.json();
}
async function apiGetQuery(path, params) {
  const url = path + '?' + new URLSearchParams(params).toString();
  const r   = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url} → ${r.status}`);
  return r.json();
}

/* ── PIPELINE TELEMETRY LOGS & TERMINAL CONSOLE ────────────── */
async function pollPipelineLogs() {
  try {
    const resp = await apiGet(`/api/pipeline/logs?since=${STATE.pipelineLogIndex || 0}`);
    if (resp && resp.logs && resp.logs.length > 0) {
      const out = document.getElementById('pipeline-terminal-output');
      const loadLine = document.getElementById('loading-log-line');
      resp.logs.forEach(log => {
        STATE.pipelineLogIndex++;
        if (out) {
          const div = document.createElement('div');
          div.className = `terminal-line stage-${log.stage} level-${log.level}`;
          div.innerHTML = `<span class="t-ts">[${log.timestamp}]</span> <span class="t-stage">[${log.stage}]</span> ${log.message}`;
          out.appendChild(div);
          out.scrollTop = out.scrollHeight;
        }
        if (loadLine) {
          loadLine.textContent = `[${log.stage}] ${log.message}`;
        }
      });
    }
  } catch(e) { /* silent */ }
}

function togglePipelineModal() {
  const el = document.getElementById('pipeline-modal-overlay');
  if (el) {
    el.classList.toggle('visible');
    if (el.classList.contains('visible')) {
      pollPipelineLogs();
    }
  }
}

function handlePipelineOverlayClick(e) {
  if (e.target && e.target.id === 'pipeline-modal-overlay') {
    togglePipelineModal();
  }
}

function clearPipelineConsole() {
  const out = document.getElementById('pipeline-terminal-output');
  if (out) {
    out.innerHTML = '<div class="terminal-line"><span class="t-ts">[00:00:00]</span> <span class="t-stage">[SYSTEM]</span> Console cleared.</div>';
  }
  STATE.pipelineLogIndex = 0;
}

/* ── NAVIGATION ────────────────────────────────────────────── */
function navigateTo(page) {
  document.querySelectorAll('.sg-page').forEach(p  => p.classList.remove('active'));
  document.querySelectorAll('.sg-nav-link').forEach(b => b.classList.remove('active'));
  const pageEl = document.getElementById('page-' + page);
  const navEl  = document.getElementById('nav-'  + page);
  if (pageEl) pageEl.classList.add('active');
  if (navEl)  navEl.classList.add('active');
  if (page === 'explorer' && STATE.map) {
    setTimeout(() => { STATE.map.invalidateSize(); }, 80);
    if (STATE.inspectedStation || STATE.selectedStation) {
      updateInspector(STATE.inspectedStation || STATE.selectedStation);
    }
  }
}

/* ── LOADING OVERLAY ───────────────────────────────────────── */
function showLoading(stage, detail, step) {
  document.getElementById('loading-stage').textContent  = stage  || 'Loading…';
  document.getElementById('loading-detail').textContent = detail || '';
  [1,2,3].forEach(i => {
    const el = document.getElementById('lstep-'+i);
    if (el) {
      el.classList.toggle('active', i === step);
      el.classList.toggle('done',   i < (step||1));
    }
  });
  document.getElementById('loading-overlay').classList.add('visible');

  // Start fast telemetry polling during loading
  if (STATE.pipelineInterval) clearInterval(STATE.pipelineInterval);
  pollPipelineLogs();
  STATE.pipelineInterval = setInterval(pollPipelineLogs, 250);
}
function hideLoading() {
  document.getElementById('loading-overlay').classList.remove('visible');
  if (STATE.pipelineInterval) {
    clearInterval(STATE.pipelineInterval);
    STATE.pipelineInterval = null;
  }
  // Flush final log entries
  pollPipelineLogs();
}

/* ── STATUS STRIP ──────────────────────────────────────────── */
function setStrip(station, state, network, step, status) {
  if (station  !== undefined) document.getElementById('strip-station').textContent = station  || '—';
  if (state    !== undefined) document.getElementById('strip-state').textContent   = state    || '—';
  if (network  !== undefined) document.getElementById('strip-network').textContent = network  || '—';
  if (step     !== undefined) document.getElementById('strip-step').textContent    = step     || '—';
  if (status   !== undefined) document.getElementById('strip-status').textContent  = status   || '—';
}

/* ================================================================
   MAP INIT — Leaflet + OpenStreetMap (100% Guaranteed, Zero Watermark)
   ================================================================ */
function initMap() {
  if (typeof L === 'undefined') {
    setTimeout(initMap, 80);
    return;
  }

  const map = L.map('sg-map', {
    center: [22.5, 79.5],
    zoom: 5,
    minZoom: 3,
    maxZoom: 18,
    zoomControl: true,
    attributionControl: false,
  });

  // OpenStreetMap standard tiles — zero watermark, no API key needed
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    subdomains: ['a', 'b', 'c'],
  }).addTo(map);

  // High-performance hardware canvas renderer
  STATE.canvasRenderer = L.canvas({ padding: 0.5 });
  STATE.dacmLayer      = L.layerGroup().addTo(map);
  STATE.windLayer      = L.layerGroup().addTo(map);
  STATE.waveLayer      = L.layerGroup().addTo(map);
  STATE.pulseLayer     = L.layerGroup().addTo(map);
  STATE.stationLayer   = L.layerGroup().addTo(map);

  STATE.map = map;

  loadCatalog();
  startAlertsPolling();
  loadNowcastLayer();

  setTimeout(() => {
    map.invalidateSize();
    hideLoading();
  }, 600);

  return map;
}

/* ── RENDER STATIONS (Canvas Circle Markers) ───────────────── */
function updateStationsLayer() {
  if (!STATE.map || !STATE.stationLayer) return;
  STATE.stationLayer.clearLayers();

  const selId = STATE.selectedStation?.station_id;

  STATE.catalog.forEach(s => {
    const id    = s.station_id;
    const cls   = classOf(id);
    const color = STATUS_COLOR[cls] || STATUS_COLOR.UNANALYSED;
    const isSel = selId === id;
    const isSub = STATE.activeSubnet.includes(id);

    const radius = isSel ? 9 : (isSub ? 6.5 : 4.5);
    const weight = isSel ? 3 : (isSub ? 2 : 1.2);
    const stroke = isSel ? '#0284c7' : '#ffffff';

    const marker = L.circleMarker([parseFloat(s.lat), parseFloat(s.lon)], {
      renderer: STATE.canvasRenderer,
      radius: radius,
      fillColor: color,
      fillOpacity: 0.92,
      color: stroke,
      weight: weight,
    });

    // Hover tooltip
    const tipHtml = `
      <div style="text-align:center;padding:2px 4px">
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;color:#0284c7">${id}</div>
        <div style="font-size:11px;font-weight:700;color:#0f172a;margin-top:1px">${s.name || id}</div>
        <div style="font-size:10px;color:#64748b;margin-top:1px">${[s.district, s.state].filter(Boolean).join(' · ')}</div>
      </div>
    `;
    marker.bindTooltip(tipHtml, {
      direction: 'top',
      offset: [0, -6],
      className: 'sg-leaflet-tooltip',
      opacity: 0.98,
    });

    // Click handler -> opens rich popup
    marker.on('click', () => onMarkerClick(s));

    STATE.stationLayer.addLayer(marker);
  });
}

function classOf(id) { return STATE.markerResults[id] || 'UNANALYSED'; }

/* ── IMD NOWCAST LAYER — Clustered Pin Markers ─────────────── */

/**
 * Creates an SVG pin icon for a nowcast station marker.
 * @param {string} color  - hex fill colour
 * @param {boolean} pulse - whether to add a pulse ring (for high-severity)
 */
function createNowcastIcon(color, pulse = false) {
  const size = 22;
  const svgPin = `
    <svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size + 8}" viewBox="0 0 ${size} ${size + 8}">
      ${ pulse ? `<circle cx="${size/2}" cy="${size/2}" r="${size/2 + 3}" fill="${color}" opacity="0.25"/>` : '' }
      <circle cx="${size/2}" cy="${size/2}" r="${size/2 - 2}" fill="${color}" stroke="white" stroke-width="2"/>
      <circle cx="${size/2}" cy="${size/2}" r="4" fill="white" opacity="0.7"/>
      <line x1="${size/2}" y1="${size - 2}" x2="${size/2}" y2="${size + 7}" stroke="${color}" stroke-width="2.5" stroke-linecap="round"/>
    </svg>`;
  return L.divIcon({
    html: svgPin,
    className: 'nowcast-pin-icon',
    iconSize:  [size, size + 8],
    iconAnchor:[size/2, size + 7],
    popupAnchor:[0, -(size + 8)],
  });
}

/** Loads the IMD Nowcast station layer (fetches /api/nowcast/stations). */
async function loadNowcastLayer() {
  if (!STATE.map) { setTimeout(loadNowcastLayer, 300); return; }
  if (typeof L.MarkerClusterGroup === 'undefined') {
    setTimeout(loadNowcastLayer, 300); return;
  }

  // Remove existing layer
  if (STATE.nowcastLayer) {
    STATE.map.removeLayer(STATE.nowcastLayer);
    STATE.nowcastLayer = null;
  }

  const countEl = document.getElementById('nowcast-station-count');
  if (countEl) countEl.textContent = 'Loading live data…';

  try {
    const resp = await apiGet('/api/nowcast/stations');
    const stations = resp.stations || [];

    // Build cluster group — spiderfies at zoom >= 9, clusters below
    const clusterGroup = L.markerClusterGroup({
      maxClusterRadius: 40,
      spiderfyOnMaxZoom: true,
      showCoverageOnHover: false,
      zoomToBoundsOnClick: true,
      disableClusteringAtZoom: 9,
      iconCreateFunction(cluster) {
        const markers = cluster.getAllChildMarkers();
        // Determine worst warning color in cluster
        let worst = 1;
        markers.forEach(m => { if ((m.options._nowcastColor||1) > worst) worst = m.options._nowcastColor||1; });
        const colors = { 1:'#16a34a', 2:'#ca8a04', 3:'#ea580c', 4:'#dc2626' };
        const bg = colors[worst] || '#16a34a';
        const cnt = cluster.getChildCount();
        const sz  = cnt > 100 ? 44 : cnt > 30 ? 36 : 28;
        return L.divIcon({
          html: `<div class="nowcast-cluster-bubble" style="width:${sz}px;height:${sz}px;background:${bg};font-size:${cnt>99?10:12}px">${cnt}</div>`,
          className: 'nowcast-cluster-icon',
          iconSize: [sz, sz],
          iconAnchor: [sz/2, sz/2],
        });
      },
    });

    stations.forEach(s => {
      const lat = parseFloat(s.lat);
      const lon = parseFloat(s.lon);
      if (isNaN(lat) || isNaN(lon)) return;

      const colorCode = s.color || 1;
      const markerColor = s.marker_color || '#16a34a';
      const isPulse = colorCode >= 4;
      const icon = createNowcastIcon(markerColor, isPulse);

      const marker = L.marker([lat, lon], { icon, _nowcastColor: colorCode });

      // Format time display
      const toiStr  = s.toi  ? `${s.toi.slice(0,2)}:${s.toi.slice(2)||'00'}`  : '—';
      const vuptoStr= s.vupto? `${s.vupto.slice(0,2)}:${s.vupto.slice(2)||'00'}`: '—';
      const msgHtml = s.message ? `<div class="ncp-message">${s.message}</div>` : '';
      const warnLabel = s.warning_level || 'No Warning';

      const popupHtml = `
        <div class="nowcast-popup-card">
          <div class="ncp-header" style="border-left: 4px solid ${markerColor}">
            <div class="ncp-id">IMD-${s.imd_id}</div>
            <div class="ncp-name">${s.name}</div>
            <span class="ncp-badge" style="background:${markerColor}">${warnLabel}</span>
          </div>
          ${msgHtml}
          <div class="ncp-meta">
            <span>🕐 Issued: ${toiStr}</span>
            <span>🕐 Valid: ${vuptoStr}</span>
            <span>📅 ${s.date}</span>
          </div>
          <div class="ncp-coords">📍 ${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E</div>
          <div class="ncp-actions">
            <div class="ncp-hint">Select analysis mode to begin SkyGuard QC:</div>
            <button class="ncp-btn ncp-btn-real"
              onclick="openNowcastStationModal(${s.imd_id}, '${s.name.replace(/'/g,'\\&apos;')}', ${lat}, ${lon}, 'REAL')">
              <i class='fa-solid fa-database'></i> Real Data
            </button>
            <button class="ncp-btn ncp-btn-synth"
              onclick="openNowcastStationModal(${s.imd_id}, '${s.name.replace(/'/g,'\\&apos;')}', ${lat}, ${lon}, 'SYNTHETIC_BENCHMARK')">
              <i class='fa-solid fa-flask-vial'></i> Benchmark
            </button>
          </div>
        </div>`;

      marker.bindPopup(popupHtml, { maxWidth: 300, closeButton: true, offset: [0, -5] });
      clusterGroup.addLayer(marker);
    });

    STATE.nowcastLayer = clusterGroup;
    if (STATE.nowcastVisible) {
      STATE.map.addLayer(clusterGroup);
    }

    if (countEl) countEl.textContent = `${stations.length} IMD stations`;
    console.log(`[Nowcast] Loaded ${stations.length} IMD stations.`);

  } catch(e) {
    console.error('[Nowcast] Failed to load nowcast stations:', e);
    if (countEl) countEl.textContent = 'Offline — using catalog';
  }
}

/** Toggle the IMD nowcast layer on/off */
function toggleNowcastLayer() {
  STATE.nowcastVisible = !STATE.nowcastVisible;
  const btn  = document.getElementById('nowcast-toggle-btn');
  const dot  = document.getElementById('nowcast-toggle-dot');
  if (STATE.nowcastVisible) {
    if (STATE.nowcastLayer && STATE.map) STATE.map.addLayer(STATE.nowcastLayer);
    if (btn) btn.classList.add('active');
  } else {
    if (STATE.nowcastLayer && STATE.map) STATE.map.removeLayer(STATE.nowcastLayer);
    if (btn) btn.classList.remove('active');
  }
}

/**
 * Called from nowcast popup buttons.
 * Finds or synthesises a catalog entry for an IMD nowcast station and opens the mode modal.
 */
function openNowcastStationModal(imdId, stationName, lat, lon, mode) {
  // Close the popup first
  if (STATE.map) STATE.map.closePopup();

  // Try to find matching entry in SkyGuard catalog (by name or coords proximity)
  let catalogEntry = STATE.catalog.find(s =>
    s.name && s.name.toLowerCase() === stationName.toLowerCase()
  );
  if (!catalogEntry) {
    // Try coordinate proximity match (within ~0.1 degree)
    catalogEntry = STATE.catalog.find(s => {
      const dLat = Math.abs(parseFloat(s.lat) - lat);
      const dLon = Math.abs(parseFloat(s.lon) - lon);
      return dLat < 0.12 && dLon < 0.12;
    });
  }

  if (catalogEntry) {
    // Use matched SkyGuard station
    openModeModal(catalogEntry, mode);
  } else {
    // Synthesise a minimal catalog entry for this IMD station
    const synth = {
      station_id: `IMD-${String(imdId).padStart(4,'0')}`,
      name:       stationName,
      state:      '',
      district:   '',
      network:    'IMD Nowcast',
      lat:        lat,
      lon:        lon,
      elevation:  null,
    };
    // Check if this IMD ID exists in ALL_INDIA_STATIONS equivalent on frontend catalog
    const byId = STATE.catalog.find(s => s.station_id === synth.station_id);
    openModeModal(byId || synth, mode);
  }
}

/* ── DACM DIRECTED ARROWS (Edge-Status Aware) ───────────────── */

/**
 * Edge status → CSS class, arrow colour, arrow unicode char.
 */
function dacmEdgeStyle(edgeStatus) {
  switch (edgeStatus) {
    case 'DOWNSTREAM':      return { cls: 'dacm-downstream-line',      arrowCls: 'dacm-arrow-downstream', arrow: '▶', color: '#0ea5e9', show: true };
    case 'WEAK_DOWNSTREAM': return { cls: 'dacm-weak-downstream-line',  arrowCls: 'dacm-arrow-weak',       arrow: '▷', color: '#38bdf8', show: true };
    case 'CROSSWIND':       return { cls: 'dacm-crosswind-line',        arrowCls: 'dacm-arrow-crosswind',  arrow: '↔', color: '#94a3b8', show: true };
    case 'UPSTREAM':        return { cls: 'dacm-upstream-line',         arrowCls: 'dacm-arrow-upstream',   arrow: '◀', color: '#475569', show: true };
    case 'CALM':            return { cls: 'dacm-calm-line',             arrowCls: '',                      arrow: '·', color: '#d97706', show: true };
    default:                return { cls: 'dacm-unavailable-line',      arrowCls: '',                      arrow: '',  color: '#64748b', show: true };
  }
}

/** Cardinal direction label for a bearing in degrees. */
function bearingToCardinal(deg) {
  const dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
  return dirs[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];
}

/**
 * Builds the scientific hover popup HTML for a DACM edge.
 */
function buildEdgePopupHtml(c, srcName, tgtName) {
  const es = c.edge_status || 'UNAVAILABLE';
  const ttStr = c.travel_time_minutes != null
    ? `~${c.travel_time_minutes.toFixed(0)} min`
    : (c.is_downstream ? '—' : 'N/A');
  const advStr = c.advective_speed_ms != null
    ? `${parseFloat(c.advective_speed_ms).toFixed(1)} m/s`
    : '—';
  const wsStr = c.wind_speed_ms != null && c.wind_speed_ms > 0
    ? `${parseFloat(c.wind_speed_ms).toFixed(1)} m/s`
    : '—';
  const wFromStr = c.wind_from_deg != null
    ? `${parseFloat(c.wind_from_deg).toFixed(0)}° (${bearingToCardinal(c.wind_from_deg)})`
    : '—';
  const wToStr = c.wind_toward_deg != null
    ? `${parseFloat(c.wind_toward_deg).toFixed(0)}° (${bearingToCardinal(c.wind_toward_deg)})`
    : '—';
  const rawCosStr = c.raw_alignment_cos != null
    ? parseFloat(c.raw_alignment_cos).toFixed(3)
    : '—';
  const provLabel = (c.wind_source === 'OPEN_METEO_ERA5') ? 'ERA5 Reanalysis' : (c.wind_source === 'STATION_OBSERVATION') ? 'Station Obs' : 'N/A';

  return `
    <div class="dacm-edge-popup">
      <div class="ep-header">
        <span>${srcName} → ${tgtName}</span>
        <span class="ep-status ${es}">${es.replace('_',' ')}</span>
      </div>
      <div class="ep-row"><span class="ep-key">Distance</span><span class="ep-val">${c.distance_km != null ? parseFloat(c.distance_km).toFixed(1) : '—'} km</span></div>
      <div class="ep-row"><span class="ep-key">Bearing (A→B)</span><span class="ep-val">${c.bearing_deg != null ? parseFloat(c.bearing_deg).toFixed(0) : '—'}° (${c.bearing_deg != null ? bearingToCardinal(c.bearing_deg) : '—'})</span></div>
      <div class="ep-row"><span class="ep-key">Wind Speed</span><span class="ep-val">${wsStr}</span></div>
      <div class="ep-row"><span class="ep-key">Wind From</span><span class="ep-val">${wFromStr}</span></div>
      <div class="ep-row"><span class="ep-key">Wind Toward</span><span class="ep-val">${wToStr}</span></div>
      <div class="ep-row"><span class="ep-key">Advective Comp.</span><span class="ep-val">${advStr}</span></div>
      <div class="ep-row"><span class="ep-key">Alignment (cos θ)</span><span class="ep-val">${rawCosStr}</span></div>
      <div class="ep-row"><span class="ep-key">Travel Time τ</span><span class="ep-val">${ttStr}</span></div>
      <div class="ep-row"><span class="ep-key">DACM Weight</span><span class="ep-val">${c.effective_weight != null ? parseFloat(c.effective_weight).toFixed(3) : '—'}</span></div>
      <div class="ep-prov">Wind source: ${provLabel}${c.wind_timestamp ? ' · ' + c.wind_timestamp.slice(0,16).replace('T',' ') + ' UTC' : ''}</div>
    </div>`;
}

/**
 * Renders directed DACM arrows on the map layer.
 * Uses /api/dacm/graph (authoritative) when available, falls back to couplings array.
 */
async function updateDACMArrows(couplings) {
  if (!STATE.map || !STATE.dacmLayer) return;
  STATE.dacmLayer.clearLayers();

  const src = STATE.selectedStation;
  if (!src) return;

  // Prefer authoritative /api/dacm/graph data
  let dacmEdges = couplings || [];
  try {
    const dacmResp = await apiGet('/api/dacm/graph');
    if (dacmResp && dacmResp.edges && dacmResp.edges.length > 0) {
      dacmEdges = dacmResp.edges;
      // Also render wind vector from source_wind
      if (dacmResp.source_wind) {
        renderWindVectorAtStation(src, dacmResp.source_wind, dacmEdges);
      }
    }
  } catch(e) { /* fall through to couplings array */ }

  // Fallback to station readings if source_wind was not provided in graph response
  if (STATE.windLayer && STATE.windLayer.getLayers().length === 0) {
    const r = STATE.stationReadings ? STATE.stationReadings[src.station_id] : null;
    if (r && r.wind_speed != null && r.wind_speed > 0.1) {
      const fallbackWind = {
        speed_ms: parseFloat(r.wind_speed),
        from_deg: parseFloat(r.wind_direction || 0),
        toward_deg: (parseFloat(r.wind_direction || 0) + 180) % 360,
        source: 'STATION_OBSERVATION',
      };
      renderWindVectorAtStation(src, fallbackWind, dacmEdges);
    }
  }

  if (!dacmEdges || dacmEdges.length === 0) return;

  const srcLatLng = [parseFloat(src.lat), parseFloat(src.lon)];
  const srcName   = src.name || src.station_id;

  dacmEdges.forEach(c => {
    const tid = c.target || c.neighbor_id || c.station_id;
    const tgt = STATE.catalog.find(s => s.station_id === tid);
    if (!tgt) return;

    const tgtLatLng = [parseFloat(tgt.lat), parseFloat(tgt.lon)];
    const tgtName   = tgt.name || tid;
    const style     = dacmEdgeStyle(c.edge_status);
    if (!style.show) return;

    // Main directed line
    const line = L.polyline([srcLatLng, tgtLatLng], {
      className: style.cls,
      color:     style.color,
      weight:    (c.edge_status === 'DOWNSTREAM') ? 3 : (c.edge_status === 'WEAK_DOWNSTREAM') ? 2 : 1.5,
    });

    // Edge hover popup
    const popupHtml = buildEdgePopupHtml(c, srcName, tgtName);
    line.bindPopup(popupHtml, { maxWidth: 300, className: '' });
    line.bindTooltip(
      `<span style="font-size:11px;font-weight:600">${c.edge_status?.replace('_',' ')} · ${tgtName}</span>`,
      { sticky: true, direction: 'auto', offset: [0, -4] }
    );

    STATE.dacmLayer.addLayer(line);

    // Arrowhead at midpoint for DOWNSTREAM / WEAK_DOWNSTREAM / CROSSWIND
    // SVG arrowhead at 65% along the line for DOWNSTREAM / WEAK_DOWNSTREAM
    if (c.edge_status === 'DOWNSTREAM' || c.edge_status === 'WEAK_DOWNSTREAM') {
      const t = 0.65;
      const midLat = srcLatLng[0] + (tgtLatLng[0] - srcLatLng[0]) * t;
      const midLon = srcLatLng[1] + (tgtLatLng[1] - srcLatLng[1]) * t;

      // Bearing in degrees (clockwise from North)
      const dLat = tgtLatLng[0] - srcLatLng[0];
      const dLon = tgtLatLng[1] - srcLatLng[1];
      const rotDeg = Math.atan2(dLon, dLat) * 180 / Math.PI;

      const arrowSize  = c.edge_status === 'DOWNSTREAM' ? 28 : 22;
      const arrowColor = style.color;
      const strokeW    = c.edge_status === 'DOWNSTREAM' ? 2 : 1.5;

      // Proper filled SVG triangle arrowhead
      const svgArrow = `
        <svg xmlns="http://www.w3.org/2000/svg"
             width="${arrowSize}" height="${arrowSize}"
             viewBox="-14 -14 28 28"
             style="transform:rotate(${rotDeg}deg);overflow:visible;display:block">
          <polygon points="0,-11 8,7 0,2 -8,7"
            fill="${arrowColor}" stroke="rgba(0,0,0,0.5)" stroke-width="${strokeW}"
            stroke-linejoin="round"/>
        </svg>`;

      const arrowIcon = L.divIcon({
        className: '',
        html: svgArrow,
        iconSize:   [arrowSize, arrowSize],
        iconAnchor: [arrowSize / 2, arrowSize / 2],
      });
      STATE.dacmLayer.addLayer(
        L.marker([midLat, midLon], { icon: arrowIcon, interactive: false })
      );
    }
  });
}

/** Backward-compat shim: updateDACMLines() now delegates to updateDACMArrows(). */
function updateDACMLines(couplings) {
  updateDACMArrows(couplings);
}

/* ── DYNAMIC WIND OVERLAY PLACEMENT OFFSET ────────────────────────── */
function computeOptimalWindBoxOffset(srcLatLng, towardDeg, fromDeg, dacmEdges, W, H) {
  const gap = 16;  // Margin in pixels from station marker

  // 8 candidate placement center offsets around the station [dx, dy]
  // angle: direction from station center in degrees (0=N, 90=E, 180=S, 270=W)
  const candidates = [
    { dir: 'W',  angle: 270, offset: L.point(-(W / 2 + gap), 0) },
    { dir: 'NW', angle: 315, offset: L.point(-Math.round((W / 2 + gap) * 0.75), -Math.round((H / 2 + gap) * 0.75)) },
    { dir: 'N',  angle: 0,   offset: L.point(0, -(H / 2 + gap)) },
    { dir: 'NE', angle: 45,  offset: L.point(Math.round((W / 2 + gap) * 0.75), -Math.round((H / 2 + gap) * 0.75)) },
    { dir: 'E',  angle: 90,  offset: L.point(W / 2 + gap, 0) },
    { dir: 'SE', angle: 135, offset: L.point(Math.round((W / 2 + gap) * 0.75), Math.round((H / 2 + gap) * 0.75)) },
    { dir: 'S',  angle: 180, offset: L.point(0, H / 2 + gap) },
    { dir: 'SW', angle: 225, offset: L.point(-Math.round((W / 2 + gap) * 0.75), Math.round((H / 2 + gap) * 0.75)) },
  ];

  // Shortest angular distance [0..180]
  const angleDiff = (a, b) => {
    const d = Math.abs(((a - b) % 360) + 360) % 360;
    return d > 180 ? 360 - d : d;
  };

  // Extract edge obstacle bearings and weights
  const edgeObstacles = [];
  if (Array.isArray(dacmEdges) && srcLatLng) {
    dacmEdges.forEach(c => {
      const tid = c.target || c.neighbor_id || c.station_id;
      const tgt = STATE.catalog ? STATE.catalog.find(s => s.station_id === tid) : null;
      let bearing = c.bearing_deg;
      if (bearing == null && tgt && tgt.lat && tgt.lon) {
        const dLat = parseFloat(tgt.lat) - srcLatLng[0];
        const dLon = parseFloat(tgt.lon) - srcLatLng[1];
        bearing = ((Math.atan2(dLon, dLat) * 180 / Math.PI) + 360) % 360;
      }
      if (bearing != null) {
        let weight = 2.0;
        const es = c.edge_status;
        if (es === 'DOWNSTREAM') weight = 6.0;
        else if (es === 'WEAK_DOWNSTREAM') weight = 4.0;
        else if (es === 'CROSSWIND') weight = 2.5;
        else if (es === 'UPSTREAM') weight = 1.8;
        edgeObstacles.push({ bearing: bearing, weight: weight });
      }
    });
  }

  // Score candidates: penalize wind propagation path & DACM lines, reward upstream side
  let bestCandidate = candidates[0];
  let maxScore = -Infinity;

  candidates.forEach(c => {
    let score = 0;

    // 1. Wind toward penalty (DO NOT place in wind forward propagation direction)
    const diffToward = angleDiff(c.angle, towardDeg);
    if (diffToward < 90) {
      const rad = diffToward * Math.PI / 180;
      score -= 5.0 * Math.cos(rad);
    }

    // 2. Wind from bonus (reward placing on upstream / from side)
    const diffFrom = angleDiff(c.angle, fromDeg);
    if (diffFrom < 90) {
      const rad = diffFrom * Math.PI / 180;
      score += 2.5 * Math.cos(rad);
    }

    // 3. DACM edge lines obstacle penalty
    edgeObstacles.forEach(obs => {
      const diffEdge = angleDiff(c.angle, obs.bearing);
      if (diffEdge < 75) {
        const rad = diffEdge * Math.PI / 180;
        const overlap = Math.pow(Math.cos(rad), 2);
        score -= obs.weight * overlap * 3.5;
      }
    });

    if (score > maxScore) {
      maxScore = score;
      bestCandidate = c;
    }
  });

  return bestCandidate.offset;
}

/* ── WIND VECTOR INDICATOR AT SELECTED STATION ──────────────── */
function renderWindVectorAtStation(station, wind, dacmEdges) {
  if (!STATE.map || !STATE.windLayer) return;
  STATE.windLayer.clearLayers();
  if (!station || !wind || !wind.speed_ms || wind.speed_ms < 0.1) return;

  const latLng    = [parseFloat(station.lat), parseFloat(station.lon)];
  const towardDeg = wind.toward_deg || ((wind.from_deg + 180) % 360);
  const cardFrom  = bearingToCardinal(wind.from_deg);
  const cardTo    = bearingToCardinal(towardDeg);
  const srcLabel  = wind.source === 'OPEN_METEO_ERA5' ? 'ERA5' : 'Obs';
  const srcColor  = wind.source === 'OPEN_METEO_ERA5' ? '#c4b5fd' : '#6ee7b7';
  const spd       = wind.speed_ms;

  const bft = spd < 0.3 ? 'Calm'
            : spd < 1.6 ? 'Light Air'
            : spd < 3.4 ? 'Light Breeze'
            : spd < 5.5 ? 'Gentle Breeze'
            : spd < 8.0 ? 'Moderate'
            : spd < 10.8 ? 'Fresh'
            : spd < 13.9 ? 'Strong'
            : 'Gale+';

  // ── Exact pixel layout ─────────────────────────────────────────
  const W = 190, H = 128;
  const divX = 91;          // divider x
  const cx = 143, cy = 66;  // compass centre (absolute)
  const R  = 30;             // compass radius

  // Calculate box position: check if user previously dragged this station's HUD
  STATE.windBoxPositions = STATE.windBoxPositions || {};
  let boxLatLng = STATE.windBoxPositions[station.station_id];

  if (!boxLatLng) {
    // Dynamically compute clearance offset from station
    const offsetPx = computeOptimalWindBoxOffset(latLng, towardDeg, wind.from_deg, dacmEdges, W, H);
    const stationPt = STATE.map.latLngToLayerPoint(latLng);
    const boxPt = stationPt.add(offsetPx);
    boxLatLng = STATE.map.layerPointToLatLng(boxPt);
  }

  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" style="display:block;cursor:grab;user-select:none;-webkit-user-select:none;pointer-events:auto;" class="wind-hud-svg">
  <!-- Outer Glow Filter -->
  <defs>
    <filter id="hud-glow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#000" flood-opacity="0.6"/>
    </filter>
  </defs>

  <!-- Panel background -->
  <rect x="0" y="0" width="${W}" height="${H}" rx="8" ry="8"
        fill="rgba(6,10,20,0.96)" stroke="rgba(251,191,36,0.9)" stroke-width="1.5"
        filter="url(#hud-glow)" pointer-events="all"/>

  <!-- Drag Header Bar (visual grip handle) -->
  <rect x="0" y="0" width="${W}" height="20" rx="8" ry="8" fill="rgba(255,255,255,0.05)" pointer-events="none"/>
  
  <!-- Grip Dots Left -->
  <circle cx="9" cy="9" r="1.2" fill="#94a3b8" opacity="0.8" pointer-events="none"/>
  <circle cx="9" cy="13" r="1.2" fill="#94a3b8" opacity="0.8" pointer-events="none"/>
  <circle cx="13" cy="9" r="1.2" fill="#94a3b8" opacity="0.8" pointer-events="none"/>
  <circle cx="13" cy="13" r="1.2" fill="#94a3b8" opacity="0.8" pointer-events="none"/>

  <!-- Grip Dots Right -->
  <circle cx="${W-13}" cy="9" r="1.2" fill="#94a3b8" opacity="0.8" pointer-events="none"/>
  <circle cx="${W-13}" cy="13" r="1.2" fill="#94a3b8" opacity="0.8" pointer-events="none"/>
  <circle cx="${W-9}" cy="9" r="1.2" fill="#94a3b8" opacity="0.8" pointer-events="none"/>
  <circle cx="${W-9}" cy="13" r="1.2" fill="#94a3b8" opacity="0.8" pointer-events="none"/>

  <!-- TITLE -->
  <text x="${W/2}" y="13.5" text-anchor="middle"
        font-family="monospace" font-size="8.5" font-weight="700" letter-spacing="1.5"
        fill="#cbd5e1" pointer-events="none">WIND TRANSPORT</text>
  <line x1="6" y1="20" x2="${W-6}" y2="20"
        stroke="rgba(148,163,184,0.22)" stroke-width="1" pointer-events="none"/>

  <!-- ── LEFT COLUMN ── -->
  <!-- Speed -->
  <text x="${divX/2}" y="52" text-anchor="middle"
        font-family="'JetBrains Mono',monospace" font-size="26" font-weight="800"
        fill="#fbbf24" pointer-events="none">${spd.toFixed(1)}</text>

  <!-- m/s -->
  <text x="${divX/2}" y="65" text-anchor="middle"
        font-family="monospace" font-size="12" font-weight="700"
        fill="#e2e8f0" pointer-events="none">m/s</text>

  <!-- Beaufort -->
  <text x="${divX/2}" y="77" text-anchor="middle"
        font-family="sans-serif" font-size="10"
        fill="#cbd5e1" pointer-events="none">${bft}</text>

  <!-- Source badge -->
  <rect x="6" y="83" width="${divX-12}" height="12" rx="3"
        fill="rgba(148,163,184,0.12)" pointer-events="none"/>
  <text x="${(divX)/2}" y="92" text-anchor="middle"
        font-family="sans-serif" font-size="9" font-weight="600"
        fill="${srcColor}" pointer-events="none">${srcLabel}</text>

  <!-- FROM label -->
  <text x="6" y="104" font-family="monospace" font-size="9"
        fill="#94a3b8" pointer-events="none">FROM</text>

  <!-- FROM bearing value -->
  <text x="6" y="118" font-family="'JetBrains Mono',monospace"
        font-size="13" font-weight="800" fill="#ffffff" pointer-events="none">${cardFrom} ${Math.round(wind.from_deg)}°</text>

  <!-- ── Vertical divider ── -->
  <line x1="${divX+2}" y1="22" x2="${divX+2}" y2="${H-8}"
        stroke="rgba(148,163,184,0.18)" stroke-width="1" pointer-events="none"/>

  <!-- ── RIGHT COLUMN — COMPASS ROSE ONLY ── -->
  <g transform="translate(${cx},${cy})" pointer-events="none">
    <!-- Outer ring -->
    <circle cx="0" cy="0" r="${R}"
            fill="rgba(255,255,255,0.03)"
            stroke="rgba(148,163,184,0.30)" stroke-width="1.5"/>

    <!-- N tick + label -->
    <line x1="0" y1="-${R}" x2="0" y2="-${R-7}"
          stroke="#e2e8f0" stroke-width="2"/>
    <text x="0" y="-${R+5}" text-anchor="middle"
          font-family="sans-serif" font-size="10" font-weight="800"
          fill="#ffffff">N</text>

    <!-- S / E / W ticks -->
    <line x1="0"     y1="${R}"  x2="0"     y2="${R-5}" stroke="rgba(148,163,184,0.35)" stroke-width="1"/>
    <line x1="-${R}" y1="0"    x2="-${R-5}" y2="0"    stroke="rgba(148,163,184,0.35)" stroke-width="1"/>
    <line x1="${R}"  y1="0"    x2="${R-5}" y2="0"     stroke="rgba(148,163,184,0.35)" stroke-width="1"/>

    <!-- Rotating wind transport arrow -->
    <g transform="rotate(${towardDeg})">
      <!-- Tail barbs -->
      <line x1="0" y1="${R*0.65}" x2="-6" y2="${R*0.65+9}"
            stroke="#fbbf24" stroke-width="2" stroke-linecap="round"/>
      <line x1="0" y1="${R*0.65}" x2="6"  y2="${R*0.65+9}"
            stroke="#fbbf24" stroke-width="2" stroke-linecap="round"/>
      <!-- Shaft -->
      <line x1="0" y1="${R*0.65}" x2="0" y2="-${R*0.72}"
            stroke="#fbbf24" stroke-width="3" stroke-linecap="round"/>
      <!-- Arrowhead -->
      <polygon points="0,-${R*0.94} 7,-${R*0.65} -7,-${R*0.65}"
               fill="#fbbf24" stroke="rgba(0,0,0,0.5)" stroke-width="1"
               stroke-linejoin="round"/>
    </g>

    <!-- Centre pivot dot -->
    <circle cx="0" cy="0" r="3" fill="#fbbf24"/>
  </g>

  <!-- TOWARD row in RIGHT column -->
  <text x="${W-6}" y="118" text-anchor="end"
        font-family="'JetBrains Mono',monospace" font-size="11" font-weight="700"
        fill="#e2e8f0" pointer-events="none">→${cardTo} ${Math.round(towardDeg)}°</text>
</svg>`;

  // Tether Leader Line connecting AWS station to HUD box
  const tetherLine = L.polyline([latLng, boxLatLng], {
    color: '#fbbf24',
    weight: 1.5,
    opacity: 0.65,
    dashArray: '4, 4',
    className: 'wind-tether-line',
    interactive: false,
  });
  STATE.windLayer.addLayer(tetherLine);

  const windIcon = L.divIcon({
    className: 'wind-transport-hud-marker',
    html: svg,
    iconSize:   [W, H],
    iconAnchor: [W / 2, H / 2],
  });

  const windMarker = L.marker(boxLatLng, {
    icon: windIcon,
    draggable: true,
    interactive: true,
    zIndexOffset: 850,
  });

  windMarker.bindTooltip(
    `<b>Wind transport · ${spd.toFixed(1)} m/s</b><br>` +
    `From ${cardFrom} ${Math.round(wind.from_deg)}° &nbsp;→&nbsp; Toward ${cardTo} ${Math.round(towardDeg)}°<br>` +
    `<span style="font-size:10px;color:${srcColor}">${wind.source === 'OPEN_METEO_ERA5' ? 'ERA5 Reanalysis' : 'Station Observation'}</span><br>` +
    `<span style="font-size:9.5px;color:#cbd5e1">✦ <b>Drag</b> to reposition freely &nbsp;·&nbsp; <b>Double-click</b> to reset</span>`,
    { direction: 'top', offset: [0, -H/2 - 6] }
  );

  // Drag handlers
  windMarker.on('dragstart', function() {
    windMarker.closeTooltip();
    if (windMarker._icon) {
      windMarker._icon.classList.add('wind-hud-dragging');
    }
  });

  windMarker.on('drag', function(e) {
    const curPos = e.latlng;
    tetherLine.setLatLngs([latLng, curPos]);
  });

  windMarker.on('dragend', function(e) {
    if (windMarker._icon) {
      windMarker._icon.classList.remove('wind-hud-dragging');
    }
    const finalPos = e.target.getLatLng();
    tetherLine.setLatLngs([latLng, finalPos]);
    STATE.windBoxPositions[station.station_id] = finalPos;
  });

  // Double click resets position to optimal clearance
  windMarker.on('dblclick', function(e) {
    L.DomEvent.stopPropagation(e);
    if (STATE.windBoxPositions) {
      delete STATE.windBoxPositions[station.station_id];
    }
    renderWindVectorAtStation(station, wind, dacmEdges);
  });

  STATE.windLayer.addLayer(windMarker);
}

/* ── PROPAGATING COLD FRONT MAP WAVE ANIMATION ──────────────── */
function updatePropagatingFrontWaves(isGenuineEvent, couplings) {
  if (!STATE.map || !STATE.waveLayer) return;
  STATE.waveLayer.clearLayers();

  if (!isGenuineEvent || !STATE.selectedStation) return;

  const src = STATE.selectedStation;
  const srcLatLng = [parseFloat(src.lat), parseFloat(src.lon)];

  // Create pulsing wavefront icon for target station
  const waveIcon = L.divIcon({
    className: 'front-wave-marker',
    html: '<div class="front-wave-ring"></div>',
    iconSize: [48, 48],
    iconAnchor: [24, 24],
  });
  STATE.waveLayer.addLayer(L.marker(srcLatLng, { icon: waveIcon }));

  // Add wave rings to DOWNSTREAM neighbors (use edge_status if available)
  (couplings || []).forEach(c => {
    const isDown = c.edge_status
      ? (c.edge_status === 'DOWNSTREAM' || c.edge_status === 'WEAK_DOWNSTREAM')
      : c.is_downstream;
    if (isDown) {
      const tgt = STATE.catalog.find(s => s.station_id === (c.target || c.neighbor_id || c.station_id));
      if (tgt) {
        const tgtLatLng = [parseFloat(tgt.lat), parseFloat(tgt.lon)];
        STATE.waveLayer.addLayer(L.marker(tgtLatLng, { icon: waveIcon }));
      }
    }
  });
}

/* ── PULSE RING FOR ACTIVE STATION ─────────────────────────── */
function setPulseMarker(station) {
  if (!STATE.map || !STATE.pulseLayer) return;
  STATE.pulseLayer.clearLayers();
  if (!station) return;

  const latLng = [parseFloat(station.lat), parseFloat(station.lon)];

  const pulse = L.circleMarker(latLng, {
    radius: 20,
    fillColor: '#0ea5e9',
    fillOpacity: 0.18,
    color: '#0284c7',
  });
  STATE.pulseLayer.addLayer(pulse);
}

/* ── ALERTS POLLING ─────────────────────────────────────────── */
function startAlertsPolling() {
  async function pollAlerts() {
    try {
      const alerts = await apiGet('/api/alerts');
      updateAlertBadge(alerts);
    } catch(e) { /* silent */ }
  }
  pollAlerts();
  STATE.alertsInterval = setInterval(pollAlerts, 30000);
}
function updateAlertBadge(alerts) {
  const badge = document.getElementById('nav-alert-badge');
  if (!badge) return;
  const count = (alerts||[]).filter(a => a.classification !== 'NORMAL').length;
  badge.textContent = count > 0 ? count : '';
  badge.style.display = count > 0 ? 'inline-flex' : 'none';
}

/* ── CATALOG LOAD ──────────────────────────────────────────── */
async function loadCatalog() {
  try {
    const resp = await apiGet('/api/india/all-stations');
    const raw  = resp.stations || resp || [];
    STATE.catalog = raw.map(s => ({
      station_id: s.station_id || s.id,
      name:       s.name || s.station_id || s.id,
      state:      s.state || '—',
      district:   s.district || '',
      network:    s.network || s.zone || '—',
      lat:        s.latitude  || s.lat,
      lon:        s.longitude || s.lon,
      elevation:  s.elevation_m || s.elevation || null,
    })).filter(s => s.lat && s.lon);

    updateStationsLayer();
    initSearch();
    setStrip(undefined, undefined, `${STATE.catalog.length} stations in catalog`, undefined, 'Select a station to begin analysis');

    // Auto-detect if a station is already active on the backend
    try {
      const status = await apiGet('/api/status');
      if (status && status.active_station_id) {
        const s = STATE.catalog.find(x => x.station_id === status.active_station_id);
        if (s) {
          STATE.selectedStation = s;
          STATE.inspectedStation = s;
          STATE.currentMode = status.data_mode || 'REAL';
          STATE.totalTimesteps = status.total_timesteps || 0;
          STATE.currentTimestep = status.current_timestep_index || 0;
          updateInspector(s);
          setStrip(s.station_id, s.state, s.network, `${STATE.currentTimestep + 1} / ${STATE.totalTimesteps || '—'}`, 'Active Station Loaded');
          if (STATE.totalTimesteps > 0) {
            enableTimeline(STATE.totalTimesteps);
            updateTimelineUI(STATE.currentTimestep, STATE.totalTimesteps);
          }
        }
      }
    } catch(errStatus) { /* ignore */ }
  } catch (e) {
    console.error('Catalog load error:', e);
    try {
      const subs = await apiGet('/api/stations');
      STATE.catalog = (subs || []).map(s => ({
        station_id: s.station_id || s.id,
        name:       s.name || s.station_id || s.id,
        state:      s.state,
        district:   s.district || '',
        network:    s.network || '—',
        lat:        s.latitude || s.lat,
        lon:        s.longitude || s.lon,
        elevation:  s.elevation_m || null,
      })).filter(s => s.lat && s.lon);
      updateStationsLayer();
      initSearch();
    } catch(e2) {
      console.error('Fallback catalog also failed:', e2);
    }
  } finally {
    hideLoading();
  }
}

/* ── RICH POPUP ON CLICK — Always routes through modal ──────── */
function onMarkerClick(station) {
  STATE.inspectedStation = station;
  updateInspector(station);

  const cls   = classOf(station.station_id);
  const id    = station.station_id;
  const readings = STATE.stationReadings[id] || {};

  const tempStr  = readings.temperature  != null ? `${parseFloat(readings.temperature).toFixed(1)} °C`  : '—';
  const pressStr = readings.pressure     != null ? `${parseFloat(readings.pressure).toFixed(1)} hPa`    : '—';
  const rhStr    = readings.humidity     != null ? `${parseFloat(readings.humidity).toFixed(0)} %`       : '—';
  const windStr  = readings.wind_speed   != null ? `${parseFloat(readings.wind_speed).toFixed(1)} m/s`  : '—';

  const metaLine = [station.district, station.state, station.network].filter(Boolean).join(' · ');

  const html = `<div class="sg-popup-card">
    <div class="popup-header">
      <div class="popup-id-tag">${id}</div>
      <div class="popup-station-name">${station.name || id}</div>
      <div class="popup-meta">${metaLine}</div>
    </div>
    <div class="popup-readings-grid">
      <div class="popup-reading"><span class="pr-label">Temp</span><span class="pr-val">${tempStr}</span></div>
      <div class="popup-reading"><span class="pr-label">Pressure</span><span class="pr-val">${pressStr}</span></div>
      <div class="popup-reading"><span class="pr-label">Humidity</span><span class="pr-val">${rhStr}</span></div>
      <div class="popup-reading"><span class="pr-label">Wind</span><span class="pr-val">${windStr}</span></div>
    </div>
    <div class="popup-coords">
      <span>📍 ${parseFloat(station.lat).toFixed(4)}° N, ${parseFloat(station.lon).toFixed(4)}° E</span>
      ${station.elevation != null ? `<span>⛰ ${parseFloat(station.elevation).toFixed(0)} m</span>` : ''}
    </div>
    <div class="popup-status-chip sg-status-chip ${cls}">${cls.replace(/_/g,' ')}</div>
    <div class="popup-action-row">
      <button class="popup-btn real-btn" onclick="openModeModalById('${id}','REAL')">
        <i class="fa-solid fa-database"></i> Real Data
      </button>
      <button class="popup-btn synth-btn" onclick="openModeModalById('${id}','SYNTHETIC_BENCHMARK')">
        <i class="fa-solid fa-flask-vial"></i> Benchmark
      </button>
    </div>
  </div>`;

  L.popup({ maxWidth: 320, offset: [0, -10], closeButton: true })
    .setLatLng([parseFloat(station.lat), parseFloat(station.lon)])
    .setContent(html)
    .openOn(STATE.map);
}

function openModeModalById(stationId, mode) {
  const s = STATE.catalog.find(x => x.station_id === stationId);
  if (s) openModeModal(s, mode);
}

function getScoreBarClass(val) {
  const v = parseFloat(val) || 0;
  return v > 0.65 ? 'high' : v > 0.35 ? 'med' : 'low';
}

function flyToStation(id) {
  const s = STATE.catalog.find(x => x.station_id === id);
  if (s && STATE.map) {
    STATE.inspectedStation = s;
    updateInspector(s);
    STATE.map.flyTo([parseFloat(s.lat), parseFloat(s.lon)], 10, { duration: 0.8 });
  }
}

/* ── DYNAMIC STATION INSPECTOR ─────────────────────────────── */
function updateInspector(station) {
  if (!station) {
    const emptyHtml = `
      <div class="sg-inspector-empty">
        <i class="fa-regular fa-circle-dot fa-2x"></i>
        <p>Click any station on the map or search above to begin analysis</p>
      </div>`;
    const titleEl = document.getElementById('inspector-title');
    const metaEl  = document.getElementById('inspector-meta');
    const bodyEl  = document.getElementById('inspector-body');
    if (titleEl) titleEl.textContent = 'Station Inspector';
    if (metaEl)  metaEl.textContent  = 'Select a station on the map';
    if (bodyEl)  bodyEl.innerHTML    = emptyHtml;
    return;
  }

  STATE.inspectedStation = station;
  const id = station.station_id;
  const isActiveTarget = STATE.selectedStation && STATE.selectedStation.station_id === id;
  const isSubnet = STATE.activeSubnet && STATE.activeSubnet.includes(id);

  // Title & Meta Header
  const titleEl = document.getElementById('inspector-title');
  const metaEl  = document.getElementById('inspector-meta');
  if (titleEl) titleEl.textContent = station.name || id;
  if (metaEl) {
    const tagCls = isActiveTarget ? 'active' : isSubnet ? 'subnet' : 'catalog';
    const tagTxt = isActiveTarget ? 'ACTIVE TARGET' : isSubnet ? 'DACM NEIGHBOR' : 'NETWORK';
    metaEl.innerHTML = `<span>${[station.district, station.state].filter(Boolean).join(' · ') || 'AWS Station'}</span> <span class="insp-tag ${tagCls}">${tagTxt}</span>`;
  }

  // Get current readings & status
  const readings = STATE.stationReadings[id] || {};
  const cls      = classOf(id);

  // Step decision info
  const latestRes = (STATE.latestStepResults && STATE.latestStepResults[id]) || (isActiveTarget && STATE.latestStepRaw) || null;
  const dec       = latestRes?.decision || (isActiveTarget ? STATE.latestStepDecision : null) || {};
  const confidence = dec.confidence != null ? (dec.confidence * 100).toFixed(1) + '%' : null;
  const severity   = dec.severity || null;
  const faultType  = latestRes?.fault_type || dec.fault_archetype || null;
  const explanation = dec.summary_explanation || null;

  // Wind & Beaufort
  const spd = readings.wind_speed != null ? parseFloat(readings.wind_speed) : null;
  const bft = spd == null ? '—'
            : spd < 0.3 ? 'Calm'
            : spd < 1.6 ? 'Light Air'
            : spd < 3.4 ? 'Light Breeze'
            : spd < 5.5 ? 'Gentle Breeze'
            : spd < 8.0 ? 'Moderate'
            : spd < 10.8 ? 'Fresh'
            : spd < 13.9 ? 'Strong'
            : 'Gale+';

  const windDir = readings.wind_direction != null ? `${bearingToCardinal(readings.wind_direction)} ${Math.round(readings.wind_direction)}°` : '—';

  // Format strings
  const tempStr  = readings.temperature != null ? `${parseFloat(readings.temperature).toFixed(1)}<span class="unit">°C</span>` : '<span class="null">—</span>';
  const pressStr = readings.pressure != null ? `${parseFloat(readings.pressure).toFixed(1)}<span class="unit">hPa</span>` : '<span class="null">—</span>';
  const rhStr    = readings.humidity != null ? `${parseFloat(readings.humidity).toFixed(0)}<span class="unit">%</span>` : '<span class="null">—</span>';
  const windStr  = spd != null ? `${spd.toFixed(1)}<span class="unit">m/s</span>` : '<span class="null">—</span>';

  let html = '';

  // 1. Live Telemetry Readings
  html += `
    <div class="sg-inspector-section">
      <div class="section-title">
        <span><i class="fa-solid fa-gauge-high"></i> Live Readings</span>
        ${readings.temperature != null ? '<span class="insp-live-indicator">● LIVE</span>' : '<span class="insp-dim-indicator">(Awaiting Step)</span>'}
      </div>
      <div class="insp-grid">
        <div class="insp-cell">
          <div class="insp-cell-label"><i class="fa-solid fa-temperature-half"></i> Temperature</div>
          <div class="insp-cell-val">${tempStr}</div>
        </div>
        <div class="insp-cell">
          <div class="insp-cell-label"><i class="fa-solid fa-droplet"></i> Rel. Humidity</div>
          <div class="insp-cell-val">${rhStr}</div>
        </div>
        <div class="insp-cell">
          <div class="insp-cell-label"><i class="fa-solid fa-cloud-arrow-down"></i> Pressure</div>
          <div class="insp-cell-val">${pressStr}</div>
        </div>
        <div class="insp-cell">
          <div class="insp-cell-label"><i class="fa-solid fa-wind"></i> Wind Speed</div>
          <div class="insp-cell-val">${windStr}</div>
          <div class="insp-cell-sub">${spd != null ? `${bft} · ${windDir}` : '—'}</div>
        </div>
      </div>
    </div>
  `;

  // 2. Anomaly Decision & AI Quality Verdict
  html += `
    <div class="sg-inspector-section">
      <div class="section-title"><span><i class="fa-solid fa-brain"></i> AI Quality Verdict</span></div>
      <div class="insp-verdict-card">
        <div class="insp-verdict-header">
          <span class="sg-status-chip ${cls}">${cls.replace(/_/g, ' ')}</span>
          ${confidence ? `<span class="insp-conf-badge">Conf: <b>${confidence}</b></span>` : ''}
        </div>
        ${severity ? `<div class="insp-verdict-row"><span class="lbl">Severity:</span> <span class="val ${severity.toLowerCase()}">${severity}</span></div>` : ''}
        ${faultType && faultType !== 'NONE' ? `<div class="insp-verdict-row"><span class="lbl">Fault Archetype:</span> <span class="val highlight">${faultType}</span></div>` : ''}
        ${explanation ? `<div class="insp-explanation">${explanation}</div>` : ''}
      </div>
    </div>
  `;

  // 3. Physics & Evidence Scores (if available)
  if (dec.temporal_score != null || dec.physics_score != null || dec.dacm_connectivity != null) {
    html += `
      <div class="sg-inspector-section">
        <div class="section-title"><span><i class="fa-solid fa-wave-square"></i> Physics & Evidence Scores</span></div>
        <div class="insp-score-list">
          <div class="insp-score-row">
            <span class="score-lbl">Ch1 Temporal Score:</span>
            <div class="sg-score-bar-wrap"><div class="sg-score-bar ${getScoreBarClass(dec.temporal_score)}" style="width:${Math.min(100, Math.max(0, (dec.temporal_score||0)*100))}%"></div></div>
            <span class="score-val">${(dec.temporal_score||0).toFixed(2)}</span>
          </div>
          <div class="insp-score-row">
            <span class="score-lbl">Ch2 Physics (T<sub>v</sub>, e, N):</span>
            <div class="sg-score-bar-wrap"><div class="sg-score-bar ${getScoreBarClass(dec.physics_score)}" style="width:${Math.min(100, Math.max(0, (dec.physics_score||0)*100))}%"></div></div>
            <span class="score-val">${(dec.physics_score||0).toFixed(2)}</span>
          </div>
          <div class="insp-score-row">
            <span class="score-lbl">Ch3 DACM Advection:</span>
            <div class="sg-score-bar-wrap"><div class="sg-score-bar ${getScoreBarClass(dec.dacm_connectivity)}" style="width:${Math.min(100, Math.max(0, (dec.dacm_connectivity||0)*100))}%"></div></div>
            <span class="score-val">${(dec.dacm_connectivity||0).toFixed(2)}</span>
          </div>
        </div>
      </div>
    `;
  }

  // 4. Station Metadata
  html += `
    <div class="sg-inspector-section">
      <div class="section-title"><span><i class="fa-solid fa-id-card"></i> Station Metadata</span></div>
      <table class="sg-metric-table">
        <tr><td>Station ID</td><td class="mono" style="color:var(--primary)">${id}</td></tr>
        <tr><td>Station Name</td><td>${station.name || '—'}</td></tr>
        <tr><td>District / State</td><td>${[station.district, station.state].filter(Boolean).join(', ') || '—'}</td></tr>
        <tr><td>Network / Zone</td><td>${station.network || '—'}</td></tr>
        <tr><td>Coordinates</td><td class="mono">${parseFloat(station.lat).toFixed(4)}° N, ${parseFloat(station.lon).toFixed(4)}° E</td></tr>
        <tr><td>Elevation</td><td class="mono">${station.elevation != null ? parseFloat(station.elevation).toFixed(0) + ' m AMSL' : '—'}</td></tr>
      </table>
    </div>
  `;

  // 5. Action Buttons
  html += `
    <div class="sg-inspector-section">
      <div class="section-title"><span><i class="fa-solid fa-sliders"></i> Actions & Controls</span></div>
      <div style="display:flex;flex-direction:column;gap:8px">
        ${isActiveTarget ? `
          <button class="sg-btn primary full-width" onclick="navigateTo('analysis')">
            <i class="fa-solid fa-chart-line"></i> Open Deep Analysis
          </button>
        ` : `
          <button class="sg-btn accent full-width" onclick="openModeModalById('${id}','REAL')">
            <i class="fa-solid fa-database"></i> Load & Ingest Real Data
          </button>
          <button class="sg-btn ghost full-width" onclick="openModeModalById('${id}','SYNTHETIC_BENCHMARK')">
            <i class="fa-solid fa-flask-vial"></i> Load Synthetic Benchmark
          </button>
        `}
        <button class="sg-btn ghost full-width" onclick="flyToStation('${id}')">
          <i class="fa-solid fa-crosshairs"></i> Center On Map
        </button>
      </div>
    </div>
  `;

  const bodyEl = document.getElementById('inspector-body');
  if (bodyEl) bodyEl.innerHTML = html;
}

/* ── MODE MODAL ────────────────────────────────────────────── */
function openModeModal(station, preselect) {
  STATE.pendingStation   = station;
  STATE.pendingPreselect = preselect || null;

  document.getElementById('modal-station-badge').textContent = station.station_id;
  document.getElementById('modal-station-name').textContent  = station.name || station.station_id;
  document.getElementById('modal-station-meta').textContent  =
    [station.district, station.state, station.network].filter(Boolean).join(' · ');

  const extra = document.getElementById('modal-station-extra');
  if (extra) {
    extra.textContent = [
      `${parseFloat(station.lat).toFixed(4)}° N, ${parseFloat(station.lon).toFixed(4)}° E`,
      station.elevation != null ? `${parseFloat(station.elevation).toFixed(0)} m elev` : null
    ].filter(Boolean).join(' · ');
  }

  document.getElementById('mode-modal-overlay').classList.add('visible');
}
function closeModeModal() {
  document.getElementById('mode-modal-overlay').classList.remove('visible');
  STATE.pendingStation = null;
}
function handleOverlayClick(e) {
  if (e.target.id === 'mode-modal-overlay') closeModeModal();
}
async function confirmModeSelect(mode) {
  if (!STATE.pendingStation) return;
  const station = STATE.pendingStation;
  closeModeModal();
  STATE.map.closePopup();
  await loadStation(station, mode);
}

/* ── LOAD / SELECT STATION ──────────────────────────────────── */
async function loadStation(station, mode) {
  stopPlayback();
  STATE.selectedStation  = station;
  STATE.inspectedStation = station;
  STATE.currentMode      = mode;
  STATE.currentTimestep  = 0;
  STATE.totalTimesteps   = 0;
  STATE.stationResults   = [];

  updateInspector(station);

  const isSynth = mode === 'SYNTHETIC_BENCHMARK';
  setStrip(station.station_id, station.state, station.network, '—', 'Loading data…');
  setModeBadge(isSynth);

  document.getElementById('bench-station-id').textContent = station.station_id;
  document.getElementById('run-benchmark-btn').disabled   = false;
  document.getElementById('run-audit-btn').disabled       = true;

  updateStationsLayer();
  setPulseMarker(station);

  try {
    showLoading('Fetching station data…', `${station.name||station.station_id} · ${mode}`, 1);

    const selectResp = await apiPostQuery('/api/india/select-station', {
      station_id: station.station_id,
      mode:       mode,
      days:       14
    });

    if (selectResp.error) throw new Error(selectResp.error);
    STATE.totalTimesteps = selectResp.timesteps || 0;
    STATE.activeSubnet   = selectResp.active_stations || [];

    updateStationsLayer();

    showLoading('Models training…', `Dual-channel BiGRU + DACM physics pipeline`, 2);

    const status = await apiGet('/api/status');
    STATE.totalTimesteps = status.total_timesteps || STATE.totalTimesteps || 0;

    showLoading('Preparing traversal…', `${STATE.totalTimesteps} timesteps available`, 3);

    updateSidebarStationList(station, selectResp, status);
    enableTimeline(STATE.totalTimesteps);
    document.getElementById('run-audit-btn').disabled = false;

    setStrip(station.station_id, station.state, station.network,
             `1 / ${STATE.totalTimesteps}`, 'Ready — step or play to traverse');
    hideLoading();

    await jumpToTimestep(0);
    navigateTo('analysis');
  } catch (err) {
    console.error('Load station error:', err);
    setStrip(undefined, undefined, undefined, undefined, 'Error: ' + err.message);
    hideLoading();
    alert('Station load failed: ' + err.message);
  }
}

function setModeBadge(isSynth) {
  const badge = document.getElementById('nav-mode-badge');
  const text  = document.getElementById('nav-mode-text');
  if (badge && text) {
    badge.className  = 'sg-mode-badge ' + (isSynth ? 'mode-synthetic' : 'mode-real');
    text.textContent = isSynth ? 'SYNTHETIC BENCHMARK' : 'REAL / HISTORICAL DATA';
  }
  const obs  = document.getElementById('obs-mode-badge');
  const obst = document.getElementById('obs-mode-text');
  if (obs && obst) {
    obs.className   = badge ? badge.className : '';
    obst.textContent = isSynth ? 'SYNTHETIC' : 'REAL';
  }
}

/* ── TIMELINE ──────────────────────────────────────────────── */
function enableTimeline(total) {
  const scrubber = document.getElementById('tl-scrubber');
  scrubber.max      = Math.max(0, total - 1);
  scrubber.value    = 0;
  scrubber.disabled = false;
  ['tl-reset','tl-prev','tl-play','tl-next'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = false;
  });
  updateTimelineUI(0, total);
}

function updateTimelineUI(idx, total) {
  total = total || STATE.totalTimesteps;
  document.getElementById('tl-step-label').textContent = `Step ${idx+1} / ${total}`;
  document.getElementById('tl-scrubber').value = idx;
}

function togglePlayback() { STATE.isPlaying ? stopPlayback() : startPlayback(); }
function startPlayback() {
  if (!STATE.selectedStation) return;
  STATE.isPlaying = true;
  document.getElementById('tl-play-icon').className = 'fa-solid fa-pause';
  document.getElementById('tl-play').classList.add('paused');
  STATE.playInterval = setInterval(async () => {
    if (STATE.currentTimestep + 1 >= STATE.totalTimesteps) { stopPlayback(); return; }
    await stepForward();
  }, STATE.playSpeed);
}
function stopPlayback() {
  STATE.isPlaying = false;
  clearInterval(STATE.playInterval);
  const icon = document.getElementById('tl-play-icon');
  if (icon) icon.className = 'fa-solid fa-play';
  const btn  = document.getElementById('tl-play');
  if (btn)  btn.classList.remove('paused');
}
async function stepForward() {
  if (STATE.currentTimestep + 1 >= STATE.totalTimesteps) return stopPlayback();
  await jumpToTimestep(STATE.currentTimestep + 1);
}
async function stepBack() {
  if (STATE.currentTimestep <= 0) return;
  await jumpToTimestep(STATE.currentTimestep - 1);
}
async function resetStream() {
  stopPlayback();
  await jumpToTimestep(0);
}
function setSpeed(ms) {
  STATE.playSpeed = ms;
  document.querySelectorAll('.sg-tl-speed-btn').forEach(b => b.classList.remove('active'));
  if (window.event?.target) window.event.target.classList.add('active');
  if (STATE.isPlaying) { stopPlayback(); startPlayback(); }
}
async function jumpToTimestep(idx) {
  if (!STATE.selectedStation) return;
  idx = Math.max(0, Math.min(idx, STATE.totalTimesteps - 1));
  STATE.currentTimestep = idx;
  updateTimelineUI(idx);
  document.getElementById('tl-scrubber').value = idx;
  await fetchStep(idx);
}

/* ── FETCH ONE STEP ─────────────────────────────────────────── */
async function fetchStep(idx) {
  if (!STATE.selectedStation) return;
  try {
    const resp = await apiGetQuery('/api/stream/jump', { timestep_index: idx });
    renderStepResult(resp);
    STATE.stationResults = STATE.stationResults || [];
    STATE.stationResults[idx] = resp;

    if (resp.timestamp) {
      const ts  = new Date(resp.timestamp);
      const ist = new Date(ts.getTime() + 5.5*60*60*1000);
      document.getElementById('tl-utc').textContent = ts.toISOString().slice(0,16).replace('T',' ') + ' UTC';
      document.getElementById('tl-ist').textContent = ist.toISOString().slice(0,16).replace('T',' ') + ' IST';
    }
    if (resp.total_timesteps && resp.total_timesteps > STATE.totalTimesteps) {
      STATE.totalTimesteps = resp.total_timesteps;
      document.getElementById('tl-scrubber').max = STATE.totalTimesteps - 1;
    }
    setStrip(undefined,undefined,undefined,`${idx+1} / ${STATE.totalTimesteps}`, undefined);
  } catch (e) {
    console.error('Step error:', e);
    try {
      const resp = await apiPost('/api/stream/step', null);
      renderStepResult(resp);
    } catch(e2) {
      setStrip(undefined,undefined,undefined,undefined,'Step error: '+e2.message);
    }
  }
}

/* ── RENDER STEP RESULT ─────────────────────────────────────── */
function renderStepResult(r) {
  document.getElementById('analysis-empty').style.display   = 'none';
  document.getElementById('analysis-panels').style.display  = 'block';

  const results  = r.results || r.step_results || {};
  const targetId = STATE.selectedStation?.station_id;
  const res      = results[targetId] || Object.values(results)[0] || {};

  const dec = res.decision || {};
  const obs = res.raw_observation || {};
  const cls = dec.classification || 'UNKNOWN';

  STATE.stationReadings[targetId] = {
    temperature:   obs.temperature,
    pressure:      obs.pressure,
    humidity:      obs.humidity,
    wind_speed:    obs.wind_speed,
    wind_direction:obs.wind_direction,
  };

  Object.entries(results).forEach(([sid, sres]) => {
    if (sid !== targetId) {
      const o = sres.raw_observation || {};
      STATE.stationReadings[sid] = {
        temperature:   o.temperature,
        pressure:      o.pressure,
        humidity:      o.humidity,
        wind_speed:    o.wind_speed,
        wind_direction:o.wind_direction,
      };
    }
  });

  STATE.markerResults[targetId] = cls;
  updateStationsLayer();

  // Cache latest step results for dynamic inspector rendering
  STATE.latestStepResults   = results;
  STATE.latestStepTimestamp = r.timestamp;
  STATE.latestStepDecision  = dec;
  STATE.latestStepRaw       = res;

  // Dynamically update Station Inspector drawer
  if (STATE.inspectedStation || STATE.selectedStation) {
    updateInspector(STATE.inspectedStation || STATE.selectedStation);
  }

  setStrip(undefined, undefined, undefined, undefined,
    cls !== 'NORMAL' && cls !== 'UNKNOWN'
      ? `⚠ ${cls.replace(/_/g,' ')} — ${dec.severity||''}`
      : '✓ Nominal operation');

  // ── Observation panel ──
  const og = document.getElementById('obs-grid');
  og.innerHTML = `
    <div class="sg-obs-cell"><div class="obs-label">Temperature</div>${fmtObs(obs.temperature)}</div>
    <div class="sg-obs-cell"><div class="obs-label">Pressure</div>${fmtObs(obs.pressure,'hPa')}</div>
    <div class="sg-obs-cell"><div class="obs-label">Rel. Humidity</div>${fmtObs(obs.humidity,'%')}</div>
    <div class="sg-obs-cell"><div class="obs-label">Wind Speed</div>${fmtObs(obs.wind_speed,'m/s')}</div>
    <div class="sg-obs-cell"><div class="obs-label">Wind Dir</div>${fmtObs(obs.wind_direction,'°')}</div>
    <div class="sg-obs-cell"><div class="obs-label">Quality</div><div class="obs-val" style="font-size:13px">${obs.quality_flag||'OK'}</div></div>`;

  let provText = STATE.currentMode === 'SYNTHETIC_BENCHMARK'
    ? 'Synthetic benchmark — station-specific fault injection'
    : `Real / historical — Open-Meteo API @ ${r.timestamp||''}`;
  
  if (r.data_provenance && r.data_provenance.ground_truth && r.data_provenance.ground_truth.expected_classification) {
    provText += ` | Ground Truth: ${r.data_provenance.ground_truth.expected_classification.replace(/_/g, ' ')}`;
  }
  document.getElementById('obs-prov-text').textContent = provText;

  // ── Decision panel ──
  const chip = document.getElementById('decision-chip');
  chip.className   = 'sg-status-chip ' + cls;
  chip.textContent = cls.replace(/_/g,' ');
  document.getElementById('dt-classification').textContent = cls.replace(/_/g,' ');
  document.getElementById('dt-severity').textContent       = dec.severity          || '—';
  document.getElementById('dt-confidence').textContent     = dec.confidence != null ? (dec.confidence*100).toFixed(1)+'%' : '—';
  document.getElementById('dt-fault').textContent          = res.fault_type         || '—';
  document.getElementById('dt-action').textContent         = dec.summary_explanation || '—';
  document.getElementById('decision-warmup-alert').style.display = 'none';

  // ── Channel 1: Temporal ──
  document.getElementById('level-ch1-text').textContent = dec.summary_explanation || '—';
  setBar('temporal', dec.temporal_score);
  setBar('local',    dec.local_anomaly_score);
  const attrs = dec.variable_attributions || {};
  setBar('attr-t', attrs.temperature || attrs.temperature_2m || 0);
  setBar('attr-p', attrs.pressure    || attrs.pressure_msl   || 0);
  setBar('attr-h', attrs.humidity    || 0);

  // ── Channel 2: Physics ──
  document.getElementById('level-ch2-text').textContent = dec.summary_explanation || '—';
  setBar('physics', dec.physics_score);
  const ch2 = res.channel2_diagnostics || {};
  setMonoCell('ch2-obs-t', obs.temperature,  '°C');
  setMonoCell('ch2-obs-p', obs.pressure,     'hPa');
  setMonoCell('ch2-obs-h', obs.humidity,     '%');
  setMonoCell('ch2-exp-t', ch2.expected_temperature, '°C');
  setMonoCell('ch2-exp-p', ch2.expected_pressure,    'hPa');
  setMonoCell('ch2-exp-h', ch2.expected_humidity,    '%');
  setMonoCell('ch2-r-tv',  ch2.virtual_temp_residual_k,     '');
  setMonoCell('ch2-r-vp',  ch2.vapor_pressure_residual_hpa, '');
  setMonoCell('ch2-r-ri',  ch2.ri_residual,                 '');

  // ── Channel 3: DACM ──
  const couplings = res.active_couplings || [];
  document.getElementById('level-ch3-text').textContent = dec.summary_explanation || '—';
  setBar('dacm',     dec.dacm_connectivity);
  setBar('prop',     dec.propagation_evidence);
  setBar('regional', dec.regional_mismatch);
  renderDACMCouplings(couplings);
  updateDACMLines(couplings);
  updatePropagatingFrontWaves(cls === 'GENUINE_METEOROLOGICAL_EVENT' || (dec.propagation_evidence && dec.propagation_evidence > 0.35), couplings);

  // ── Sensor health ──
  const health = res.health || {};
  document.getElementById('health-overall-chip').className   = 'sg-status-chip ' + (health.overall||'UNKNOWN');
  document.getElementById('health-overall-chip').textContent = health.overall || '—';
  document.getElementById('health-reliability').textContent  = health.reliability_score != null
    ? (health.reliability_score*100).toFixed(1)+'%'
    : (health.reliability!=null ? (health.reliability*100).toFixed(1)+'%' : '—');
  document.getElementById('health-fault-rate').textContent   = health.fault_rate != null
    ? (health.fault_rate*100).toFixed(1)+'%' : '—';
  renderHealthVars(health.variable_health || health.variables);
}

/* ── DACM TABLE ─────────────────────────────────────────────── */
function renderDACMCouplings(couplings) {
  const tbody = document.getElementById('dacm-coupling-body');
  if (!couplings || couplings.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-td">No coupling data for this timestep</td></tr>'; return;
  }

  const STATUS_STYLE = {
    DOWNSTREAM:      { badge: '↓ DOWNSTREAM',   color: '#0ea5e9', bg: 'rgba(14,165,233,0.1)' },
    WEAK_DOWNSTREAM: { badge: '↘ WEAK↓',         color: '#38bdf8', bg: 'rgba(56,189,248,0.1)' },
    CROSSWIND:       { badge: '↔ CROSSWIND',     color: '#94a3b8', bg: 'rgba(148,163,184,0.1)' },
    UPSTREAM:        { badge: '↑ UPSTREAM',      color: '#475569', bg: 'rgba(71,85,105,0.1)'  },
    CALM:            { badge: '○ CALM',           color: '#d97706', bg: 'rgba(217,119,6,0.1)'  },
    UNAVAILABLE:     { badge: '? N/A',            color: '#64748b', bg: 'rgba(100,116,139,0.1)'},
  };

  tbody.innerHTML = couplings.map(c => {
    const es   = c.edge_status || (c.is_downstream ? 'DOWNSTREAM' : 'UNAVAILABLE');
    const st   = STATUS_STYLE[es] || STATUS_STYLE.UNAVAILABLE;
    const tid  = c.target || c.neighbor_id || c.station_id || '—';
    const cs   = STATE.catalog.find(x => x.station_id === tid);
    const name = cs ? (cs.district || cs.name || tid) : tid;
    const w    = c.effective_weight ?? c.dacm_weight ?? c.weight ?? 0;

    const advStr = c.advective_speed_ms != null && c.advective_speed_ms > 0
      ? `${parseFloat(c.advective_speed_ms).toFixed(1)} m/s`
      : (c.wind_speed_ms > 0 ? `${parseFloat(c.wind_speed_ms).toFixed(1)} m/s¹` : '—');

    const ttStr = c.travel_time_minutes != null
      ? `~${Math.round(c.travel_time_minutes)} min`
      : '—';

    return `<tr>
      <td class="mono" style="color:${st.color}">
        ${tid}<br>
        <span style="font-size:10px;color:var(--text-muted)">${name}</span>
      </td>
      <td class="mono">${c.distance_km!=null?parseFloat(c.distance_km).toFixed(1):'—'}</td>
      <td class="mono">${c.bearing_deg!=null?parseFloat(c.bearing_deg).toFixed(0)+'°':'—'}</td>
      <td class="mono" style="color:${st.color}">${advStr}</td>
      <td>
        <span style="display:inline-block;padding:2px 6px;border-radius:8px;
          background:${st.bg};color:${st.color};font-size:10px;font-weight:700;white-space:nowrap">
          ${st.badge}
        </span>
      </td>
      <td class="mono">${ttStr}</td>
    </tr>`;
  }).join('');
}


/* ── HEALTH VARS TABLE ──────────────────────────────────────── */
function renderHealthVars(vars) {
  const tbody = document.getElementById('health-var-body');
  if (!vars || typeof vars !== 'object' || !Object.keys(vars).length) {
    tbody.innerHTML = '<tr><td colspan="2" class="empty-td">—</td></tr>'; return;
  }
  tbody.innerHTML = Object.entries(vars).map(([name, h]) => {
    const st = typeof h === 'object' ? (h.status||'—') : String(h);
    return `<tr><td>${name.replace(/_/g,' ')}</td>
      <td><span class="sg-status-chip ${st}" style="font-size:10px">${st}</span></td></tr>`;
  }).join('');
}

/* ── FULL PASS AUDIT ─────────────────────────────────────────── */
async function runFullPassAudit() {
  if (!STATE.selectedStation) return;
  const stationId = STATE.selectedStation.station_id;
  document.getElementById('audit-empty').style.display      = 'none';
  document.getElementById('audit-summary').style.display    = 'none';
  document.getElementById('audit-table-wrap').style.display = 'none';
  document.getElementById('audit-export-row').style.display = 'none';
  document.getElementById('audit-loading').style.display    = 'flex';
  document.getElementById('run-audit-btn').disabled = true;
  stopPlayback();
  try {
    const resp = await apiGet(`/api/stations/${stationId}/full-pass-report`);
    STATE.auditResults = resp.anomalous_points || [];
    renderAuditReport(resp);
  } catch (e) {
    console.error('Audit error:', e);
    await runAuditManually(stationId);
  } finally {
    document.getElementById('audit-loading').style.display = 'none';
    document.getElementById('run-audit-btn').disabled = false;
  }
}

async function runAuditManually(stationId) {
  const results = [];
  const total   = STATE.totalTimesteps || 0;
  for (let i = 0; i < total; i++) {
    try {
      const resp = await apiPost('/api/stream/step', null);
      const res  = (resp.results||{})[stationId] || Object.values(resp.results||{})[0] || {};
      const dec  = res.decision || {};
      results.push({
        timestamp_ist: resp.timestamp
          ? new Date(new Date(resp.timestamp).getTime()+5.5*60*60*1000).toISOString().slice(0,16).replace('T',' ')
          : '—',
        classification: dec.classification||'UNKNOWN',
        severity:       dec.severity||'—',
        confidence:     dec.confidence,
        measured: {
          temperature: res.raw_observation?.temperature,
          pressure:    res.raw_observation?.pressure,
          humidity:    res.raw_observation?.humidity,
        },
        reasoning: dec.summary_explanation||'—',
      });
    } catch(e) { results.push({ classification:'ERROR', reasoning: e.message }); }
  }
  STATE.auditResults = results;
  renderAuditReport({ anomalous_points: results.filter(r => r.classification !== 'NORMAL' && r.classification !== 'UNKNOWN') });
}

function renderAuditReport(resp) {
  const points = resp.anomalous_points || STATE.auditResults || [];

  const summaryEl = document.getElementById('audit-summary');
  summaryEl.style.display = 'grid';

  if (resp.total_timesteps_audited != null) {
    summaryEl.innerHTML = `
      <div class="sg-obs-cell" style="text-align:center">
        <div class="obs-label">Timesteps</div>
        <div class="obs-val">${resp.total_timesteps_audited}</div>
      </div>
      <div class="sg-obs-cell" style="text-align:center">
        <div class="obs-label">Anomalies</div>
        <div class="obs-val" style="color:var(--c-fault)">${resp.total_anomalies_detected||0}</div>
      </div>
      <div class="sg-obs-cell" style="text-align:center">
        <div class="obs-label">Quality</div>
        <div class="obs-val" style="color:var(--c-normal)">${resp.data_quality_percentage||'—'}%</div>
      </div>
      <div class="sg-obs-cell" style="text-align:center;grid-column:span 3">
        <div class="obs-label">Verdict</div>
        <div class="obs-val" style="font-size:13px;color:${resp.overall_verdict==='NOMINAL_PHYSICAL_OPERATION'?'var(--c-normal)':'var(--c-fault)'}">
          ${(resp.overall_verdict||'—').replace(/_/g,' ')}
        </div>
      </div>`;
  } else {
    const counts = {};
    points.forEach(p => { const c = p.classification||'UNKNOWN'; counts[c]=(counts[c]||0)+1; });
    summaryEl.innerHTML = Object.entries(counts).map(([c,n]) =>
      `<div class="sg-obs-cell" style="text-align:center">
        <div class="obs-label">${c.replace(/_/g,' ')}</div>
        <div class="obs-val" style="color:${STATUS_COLOR[c]||'var(--text-muted)'}">${n}</div>
      </div>`).join('');
  }

  const tbody = document.getElementById('audit-table-body');
  tbody.innerHTML = points.map(p => {
    const cls  = p.classification || '—';
    const chip = `<span class="sg-status-chip ${cls}" style="font-size:9px">${cls.replace(/_/g,' ')}</span>`;
    const m    = p.measured || {};
    return `<tr>
      <td class="mono" style="font-size:11px">${p.timestamp_ist||p.timestamp_utc||'—'}</td>
      <td>${chip}</td>
      <td>${p.severity||'—'}</td>
      <td class="mono">${p.confidence!=null?(parseFloat(p.confidence)*100).toFixed(0)+'%':'—'}</td>
      <td class="mono">${m.temperature!=null?parseFloat(m.temperature).toFixed(1):'—'}</td>
      <td class="mono">${m.pressure!=null?parseFloat(m.pressure).toFixed(1):'—'}</td>
      <td class="mono">${m.humidity!=null?parseFloat(m.humidity).toFixed(0):'—'}</td>
      <td style="font-size:11px;max-width:200px;word-break:break-word">${trunc(p.reasoning||'—',90)}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="8" class="empty-td">No anomalous points detected — station operating nominally.</td></tr>';

  document.getElementById('audit-table-wrap').style.display = 'block';
  document.getElementById('audit-export-row').style.display = 'block';
}

function exportAuditJSON() {
  if (!STATE.auditResults) return;
  const blob = new Blob([JSON.stringify({ station:STATE.selectedStation?.station_id, mode:STATE.currentMode, results:STATE.auditResults },null,2)],{type:'application/json'});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `skyguard_audit_${STATE.selectedStation?.station_id||'unknown'}_${Date.now()}.json`;
  a.click(); URL.revokeObjectURL(url);
}

/* ── BENCHMARKS ─────────────────────────────────────────────── */
async function runBenchmarks() {
  if (!STATE.selectedStation) return;
  document.getElementById('benchmark-loading').style.display = 'flex';
  document.getElementById('benchmark-empty').style.display   = 'none';
  document.getElementById('scenario-grid').innerHTML = '';
  document.getElementById('run-benchmark-btn').disabled = true;
  try {
    const resp = await apiGetQuery('/api/benchmarks/scenarios', { station_id: STATE.selectedStation.station_id });
    const scenarios = resp.scenarios || [];
    renderBenchmarkCards(scenarios);
  } catch (e) {
    console.error('Benchmark error:', e);
    document.getElementById('scenario-grid').innerHTML =
      `<div class="sg-alert warning"><i class="fa-solid fa-triangle-exclamation"></i> Benchmark error: ${e.message}</div>`;
  } finally {
    document.getElementById('benchmark-loading').style.display = 'none';
    document.getElementById('run-benchmark-btn').disabled = false;
  }
}

function renderBenchmarkCards(scenarios) {
  const grid = document.getElementById('scenario-grid');
  if (!scenarios.length) {
    grid.innerHTML = '<div class="sg-alert warning"><i class="fa-solid fa-triangle-exclamation"></i> No scenario results returned.</div>'; return;
  }
  grid.innerHTML = scenarios.map((sc, i) => {
    const pass    = sc.passed !== undefined ? sc.passed : (sc.target_status === sc.expected_status);
    const badgeClass = pass ? 'pass' : 'fail';
    const badgeText  = pass ? 'PASS' : 'FAIL';
    const exp     = sc.expected_status || sc.expected || '—';
    const det     = sc.target_status   || sc.detected || '—';
    const ePhys   = sc.e_phys != null ? parseFloat(sc.e_phys).toFixed(3) : '—';
    const eProp   = sc.e_prop != null ? parseFloat(sc.e_prop).toFixed(3) : '—';
    const expEv   = sc.expected_evidence || {};

    const expEvHtml = (expEv.temporal || expEv.physics || expEv.dacm) ? `
      <div class="sc-evidence-box" style="margin-top:6px;font-size:11px">
        ${expEv.temporal ? `<div><strong>Temporal (Ch1):</strong> ${expEv.temporal}</div>` : ''}
        ${expEv.physics  ? `<div><strong>Physics (Ch2):</strong> ${expEv.physics}</div>` : ''}
        ${expEv.dacm     ? `<div><strong>DACM (Ch3):</strong> ${expEv.dacm}</div>` : ''}
      </div>` : '';

    return `<div class="sg-scenario-card ${pass ? 'passed' : 'failed'}">
      <div class="sc-header">
        <div class="sc-title">${sc.title || sc.name || `Scenario ${i+1}`}</div>
        <span class="sc-badge ${badgeClass}">${badgeText}</span>
      </div>
      <div class="sc-desc">${sc.description || ''}</div>
      <div class="sc-evidence-box">
        <div class="sc-evidence-row">
          <span class="sc-evidence-label">Physics Inconsistency (E_phys):</span>
          <span class="sc-evidence-val" style="color:${sc.e_phys > 0.35 ? 'var(--c-fault)' : 'var(--c-normal)'}">${ePhys}</span>
        </div>
        <div class="sc-evidence-row">
          <span class="sc-evidence-label">Advective Propagation (E_prop):</span>
          <span class="sc-evidence-val" style="color:${sc.e_prop > 0.35 ? '#a855f7' : 'var(--text-muted)'}">${eProp}</span>
        </div>
      </div>
      ${expEvHtml}
      <div class="sc-verdict-row">
        <div>
          <span style="font-size:10px;color:var(--text-muted)">Expected:</span>
          <span class="sg-status-chip ${exp}" style="font-size:9px">${exp.replace(/_/g,' ')}</span>
        </div>
        <i class="fa-solid fa-arrow-right" style="color:var(--text-muted);font-size:10px"></i>
        <div>
          <span style="font-size:10px;color:var(--text-muted)">Model Detected:</span>
          <span class="sg-status-chip ${det}" style="font-size:9px">${det.replace(/_/g,' ')}</span>
        </div>
      </div>
    </div>`;
  }).join('');
}

/* ── SEARCH ─────────────────────────────────────────────────── */
function initSearch() {
  const input   = document.getElementById('station-search-input');
  const results = document.getElementById('search-results');

  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) { results.classList.remove('visible'); return; }
    const matches = STATE.catalog.filter(s =>
      (s.station_id?.toLowerCase().includes(q)) ||
      (s.name?.toLowerCase().includes(q)) ||
      (s.state?.toLowerCase().includes(q)) ||
      (s.district?.toLowerCase().includes(q))
    ).slice(0, 12);
    results.innerHTML = matches.length
      ? matches.map(s => `
        <div class="sg-search-result-item" onclick="searchSelect('${s.station_id}')">
          <div class="stn-id">${s.station_id}</div>
          <div class="stn-name">${s.name||s.station_id}</div>
          <div class="stn-state">${[s.district,s.state,s.network].filter(Boolean).join(' · ')}</div>
        </div>`).join('')
      : '<div class="sg-search-result-item" style="color:var(--text-muted)">No stations found</div>';
    results.classList.add('visible');
  });
  document.addEventListener('click', e => {
    if (!e.target.closest('.sg-search-box')) results.classList.remove('visible');
  });
}

function searchSelect(id) {
  document.getElementById('search-results').classList.remove('visible');
  document.getElementById('station-search-input').value = '';
  const s = STATE.catalog.find(x => x.station_id === id);
  if (!s) return;
  STATE.inspectedStation = s;
  updateInspector(s);
  STATE.map.flyTo([parseFloat(s.lat), parseFloat(s.lon)], 9, { duration: 0.9 });
  setTimeout(() => onMarkerClick(s), 950);
}

/* ── SIDEBAR STATION LIST ───────────────────────────────────── */
function updateSidebarStationList(station, selectResp, status) {
  const el        = document.getElementById('sidebar-station-list');
  const neighbors = (selectResp.active_stations || []).filter(id => id !== station.station_id).slice(0, 6);

  el.innerHTML = `
    <div class="sidebar-stn-item" style="border:1px solid var(--primary);border-radius:8px;padding:8px;margin-bottom:10px">
      <div class="stn-id" style="color:var(--primary)">${station.station_id} <span style="font-size:9px;color:var(--primary-glow)">(ACTIVE)</span></div>
      <div class="stn-name" style="font-size:12px;color:var(--chrome-text)">${station.name||'—'}</div>
      <div class="stn-state">${[station.district,station.state].filter(Boolean).join(' · ')}</div>
    </div>
    <div class="section-title"><i class="fa-solid fa-link"></i> DACM Sub-Network</div>
    ${neighbors.map(id => {
      const s = STATE.catalog.find(x => x.station_id === id);
      return `<div class="sidebar-stn-item" onclick="searchSelect('${id}')">
        <div class="stn-id">${id}</div>
        <div class="stn-name" style="font-size:11px;color:var(--chrome-text)">${s?.name||'—'}</div>
        <div class="stn-state">${s ? [s.district,s.state].filter(Boolean).join(' · ') : 'Context provider'}</div>
      </div>`;
    }).join('') || '<div class="sidebar-empty">No neighbour data</div>'}`;
}

/* ── CHANNEL TAB SWITCHER ───────────────────────────────────── */
function switchChTab(id) {
  document.querySelectorAll('.sg-ch-tab').forEach(t  => t.classList.remove('active'));
  document.querySelectorAll('.sg-ch-content').forEach(c => c.classList.remove('active'));
  const t = document.getElementById('tab-'+id);
  const c = document.getElementById('ch-content-'+id);
  if (t) t.classList.add('active');
  if (c) c.classList.add('active');
}

/* ── UTILITY ────────────────────────────────────────────────── */
function fmtObs(val, unit='°C') {
  if (val == null || isNaN(val)) return '<div class="obs-null">—</div>';
  return `<div class="obs-val">${parseFloat(val).toFixed(1)}<span class="obs-unit">${unit}</span></div>`;
}
function setMonoCell(id, val, unit) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = val != null && !isNaN(val) ? parseFloat(val).toFixed(2)+(unit||'') : '—';
}
function setBar(id, val) {
  const bar  = document.getElementById('bar-'+id);
  const text = document.getElementById('val-'+id);
  if (!bar || !text) return;
  const v   = parseFloat(val) || 0;
  const pct = Math.min(100, Math.max(0, v * 100));
  bar.style.width = pct + '%';
  bar.className   = 'sg-score-bar ' + (pct > 65 ? 'high' : pct > 35 ? 'med' : 'low');
  text.textContent = val != null ? v.toFixed(3) : '—';
}
function trunc(s, n) { return s && s.length > n ? s.slice(0,n)+'…' : (s||'—'); }

/* ── BOOT ──────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  showLoading('Initialising SkyGuard AI…', 'Connecting to backend & loading map…', 1);
  initMap();
});
