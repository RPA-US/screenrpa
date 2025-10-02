/**
 * Process Discovery + Emotions Analysis (con tabs)
 * Se alinea con los IDs del HTML:
 *  - #pdActivityEmotionsChart
 *  - #pdHeatmapChart
 *  - #pdNegativeActivitiesRanking
 * Y con la estructura de datos en extraData.pd_log.process_discovery_analysis
 */

// Paleta por emoción (fallback si aparece una nueva)
const EMOTION_COLORS = {
  neutral: 'rgba(255, 206, 86, 0.7)',
  happy: 'rgba(75, 192, 192, 0.7)',
  sad: 'rgba(108, 117, 125, 0.7)',
  angry: 'rgba(255, 99, 132, 0.7)',
  surprised: 'rgba(23, 162, 184, 0.7)',
  fearful: 'rgba(111, 66, 193, 0.7)',
  disgusted: 'rgba(253, 126, 20, 0.7)'
};
function fallbackColor(key) {
  let hash = 0; for (let i = 0; i < key.length; i++) hash = key.charCodeAt(i) + ((hash << 5) - hash);
  const hue = Math.abs(hash) % 360;
  return `hsla(${hue},70%,55%,0.7)`;
}

// Obtiene pda desde extraData (inyectado por Django)
function getPDA() {
  const extraEl = document.getElementById('extraData');
  if (!extraEl) return null;
  try {
    const extra = JSON.parse(extraEl.textContent || '{}');
    return extra?.pd_log?.process_discovery_analysis || null;
  } catch (e) {
    console.error('No se pudo parsear extraData', e);
    return null;
  }
}

// ---------- Gráfico 1: Barras apiladas (stacked) ----------
let stackedChart = null;
function renderActivityEmotionsChart() {
  const pda = getPDA();
  const el = document.getElementById('pdActivityEmotionsChart');
  const sbd = pda?.stacked_bar_data;
  if (!el || !sbd) {
    if (el) el.parentNode.innerHTML = '<div class="alert alert-info mb-0">No hay datos suficientes para visualizar</div>';
    return;
  }

  const activities = (sbd.activities || []).map(String);
  const emotions = (sbd.emotions || []).map(String);
  const data = sbd.data || {};

  if (!activities.length || !emotions.length) {
    el.parentNode.innerHTML = '<div class="alert alert-info mb-0">No hay datos suficientes para visualizar</div>';
    return;
  }

  const datasets = emotions.map(em => {
    const color = EMOTION_COLORS[em] || fallbackColor(em);
    const border = color.replace('0.7', '1');
    return {
      label: em,
      data: activities.map(a => (data[a] && data[a][em]) ? data[a][em] : 0),
      backgroundColor: color,
      borderColor: border,
      borderWidth: 1,
      stack: 'total'
    };
  });

  stackedChart = new Chart(el.getContext('2d'), {
    type: 'bar',
    data: { labels: activities, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, title: { display: true, text: 'Actividades (ocel:activity)' } },
        y: { stacked: true, title: { display: true, text: 'Frecuencia' } }
      },
      plugins: {
        legend: { position: 'top' },
        title: { display: true, text: 'Distribución de Emociones por Actividad' },
        tooltip: {
          callbacks: {
            footer: (items) => {
              const total = items.reduce((acc, it) => acc + (it.raw || 0), 0);
              return 'Total actividad: ' + total;
            }
          }
        }
      }
    }
  });
}

// ---------- Gráfico 2: Heatmap (matrix) ----------
let heatmapChart = null;
function renderHeatmapChart() {
  const pda = getPDA();
  const el = document.getElementById('pdHeatmapChart');
  const h = pda?.heatmap_data;
  if (!el) return;
  if (!h) {
    el.parentNode.innerHTML = '<div class="alert alert-info mb-0">No hay datos de heatmap disponibles</div>';
    return;
  }

  const activities = (h.activities || []).map(String);
  const emotions = (h.emotions || []).map(String);
  const intensity = h.intensity || {};

  if (!activities.length || !emotions.length) {
    el.parentNode.innerHTML = '<div class="alert alert-info mb-0">No hay datos suficientes para el heatmap</div>';
    return;
  }

  const matrixData = [];
  let maxV = 0;
  emotions.forEach((em, x) => {
    const byAct = intensity[em] || {};
    activities.forEach((act, y) => {
      const v = Number(byAct[act] || 0);
      if (v > 0) {
        matrixData.push({ x, y, v });
        if (v > maxV) maxV = v;
      }
    });
  });

  if (maxV === 0) {
    el.parentNode.innerHTML = '<div class="alert alert-info mb-0">No hay intensidad positiva para pintar el heatmap</div>';
    return;
  }

  try {
    heatmapChart = new Chart(el.getContext('2d'), {
      type: 'matrix',
      data: {
        datasets: [{
          label: 'Intensidad Emocional',
          data: matrixData,
          backgroundColor: (ctx) => {
            const v = ctx.raw ? ctx.raw.v : 0;
            const alpha = Math.max(0.1, v / maxV);
            return `rgba(255, 99, 132, ${alpha})`;
          },
          borderWidth: 1,
          borderColor: '#fff',
          width: (ctx) => {
            const a = ctx.chart.chartArea;
            return (a.right - a.left) / emotions.length - 1;
          },
          height: (ctx) => {
            const a = ctx.chart.chartArea;
            return (a.bottom - a.top) / activities.length - 1;
          }
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Heatmap Actividad–Emoción' },
          tooltip: {
            callbacks: {
              title: (items) => {
                if (!items.length) return '';
                const d = items[0].raw;
                return `Actividad: ${activities[d.y]} · Emoción: ${emotions[d.x]}`;
              },
              label: (item) => `Conteo: ${item.raw ? item.raw.v : 0}`
            }
          }
        },
        scales: {
          x: {
            type: 'category',
            labels: emotions,
            offset: true,
            title: { display: true, text: 'Emociones' },
            ticks: { maxRotation: 0 }
          },
          y: {
            type: 'category',
            labels: activities,
            offset: true,
            title: { display: true, text: 'Actividades (ocel:activity)' }
          }
        }
      }
    });
  } catch (e) {
    el.parentNode.innerHTML = '<div class="alert alert-warning mb-0">No se pudo renderizar el heatmap. Revisa que <code>chartjs-chart-matrix@2</code> esté cargado tras Chart.js v4.</div>';
  }
}

// ---------- Ranking (tabla) ----------
let rankingRendered = false;
function renderNegativeActivitiesRanking() {
  if (rankingRendered) return;

  const pda = getPDA();
  const container = document.getElementById('pdNegativeActivitiesRanking');
  const obj = pda?.negativity_ranking;
  if (!container) return;

  if (!obj || !Object.keys(obj).length) {
    container.innerHTML = '<div class="alert alert-info mb-0">No hay datos de ranking de negatividad</div>';
    return;
  }

  const arr = Object.keys(obj).map(act => ({
    activity: String(act),
    negative_count: Number(obj[act].negative_count || 0),
    total_count: Number(obj[act].total_count || 0),
    negativity_ratio: Number(obj[act].negativity_ratio || 0)
  })).sort((a,b) => b.negativity_ratio - a.negativity_ratio);

  if (!arr.length) {
    container.innerHTML = '<div class="alert alert-info mb-0">No hay datos suficientes para el ranking</div>';
    return;
  }

  let html = `
    <div class="table-responsive">
      <table class="table table-sm">
        <thead>
          <tr>
            <th style="width:48px">#</th>
            <th>Actividad</th>
            <th style="width:140px" class="text-end">% Negatividad</th>
            <th>Proporción</th>
          </tr>
        </thead>
        <tbody>
  `;
  arr.forEach((row, i) => {
    const pct = isFinite(row.negativity_ratio) ? (row.negativity_ratio * 100).toFixed(1) : '0.0';
    html += `
      <tr class="ranking-item">
        <td>${i+1}</td>
        <td><span class="badge bg-light text-dark"> ${row.activity} </span></td>
        <td class="ranking-value text-end">${pct}%</td>
        <td>
          <div class="ranking-bar" style="width:${pct}%; max-width:100%"></div>
          <small class="text-muted">${row.negative_count} / ${row.total_count} negativos</small>
        </td>
      </tr>
    `;
  });
  html += `</tbody></table></div>`;
  container.innerHTML = html;

  rankingRendered = true;
}

// ---------- Init: render inicial + escucha de tabs ----------
(function initProcessEmotions() {
  // Pinta la pestaña activa (stacked)
  renderActivityEmotionsChart();

  // Cuando cambie de pestaña, pinta la correspondiente (Bootstrap 4/5)
  const tabs = document.querySelectorAll('#processEmotionsTabs a[data-toggle="tab"], #processEmotionsTabs a[data-bs-toggle="tab"]');
  tabs.forEach(tab => {
    tab.addEventListener('shown.bs.tab', function () {
      const target = this.getAttribute('href') || this.dataset.bsTarget || '';
      if (target === '#process-heatmap') {
        renderHeatmapChart();
      } else if (target === '#process-ranking') {
        renderNegativeActivitiesRanking();
      } else if (target === '#process-stacked') {
        // ya renderizado
      }
    });
  });
})();
