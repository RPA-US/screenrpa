document.addEventListener('DOMContentLoaded', function() {
  // Obtener los datos JSON
  var chartLabels = JSON.parse(document.getElementById('chart-labels-data').textContent);
  var chartDatasets = JSON.parse(document.getElementById('chart-datasets-data').textContent);
  var statsData = JSON.parse(document.getElementById('stats-data').textContent);
  var eventsData = JSON.parse(document.getElementById('events-data').textContent);
  var indicatorsData = JSON.parse(document.getElementById('indicators-data').textContent);
  var defaultChartType = JSON.parse(document.getElementById('chart-type-data').textContent) || 'line';
  
  console.log("Tipo de gráfico por defecto:", defaultChartType);
  
  // Crear gráficos para cada métrica
  var charts = {};
  
  // Configuración de métricas
  window.metricConfig = {
    'fc': {
      color: '#f5365c', 
      bgColor: 'rgba(245, 54, 92, 0.2)',
      displayName: 'Heart Rate (bpm)',
      unit: 'bpm'
    },
    'pasos': {
      color: '#5e72e4',
      bgColor: 'rgba(94, 114, 228, 0.2)',
      displayName: 'Steps',
      unit: ''
    },
    'calorias': {
      color: '#fb6340',
      bgColor: 'rgba(251, 99, 64, 0.2)',
      displayName: 'Calories',
      unit: 'cal'
    },
    'zona_activa': {
      color: '#2dce89',
      bgColor: 'rgba(45, 206, 137, 0.2)',
      displayName: 'Active Zone Minutes',
      unit: 'min'
    },
    'sedentario': {
      color: '#11cdef',
      bgColor: 'rgba(17, 205, 239, 0.2)',
      displayName: 'Sedentary Time',
      unit: 'min'
    },
    'ratio_fc_pasos': {
      color: '#8965e0',
      bgColor: 'rgba(137, 101, 224, 0.2)',
      displayName: 'HR/Steps Ratio',
      unit: ''
    },
    'cvl': {
      color: '#ffd600',
      bgColor: 'rgba(255, 214, 0, 0.2)',
      displayName: 'CVL',
      unit: ''
    },
    'sdnn': {
      color: '#8898aa',
      bgColor: 'rgba(136, 152, 170, 0.2)',
      displayName: 'SDNN',
      unit: 'ms'
    },
    'spo2': {
      color: '#1d8cf8',
      bgColor: 'rgba(29, 140, 248, 0.2)',
      displayName: 'SpO₂',
      unit: '%'
    },
    'temperatura': {
      color: '#a38df8',
      bgColor: 'rgba(163, 141, 248, 0.2)',
      displayName: 'Temperature Variation',
      unit: '°C'
    },
    'hrv': {
      color: '#f58231',
      bgColor: 'rgba(245, 130, 49, 0.2)',
      displayName: 'HRV',
      unit: 'ms'
    }
  };
  
  // Función para formatear valores específicos según la métrica
  function formatMetricValue(value, metric) {
    if (metric === 'temperatura') {
      // Para temperatura, mostrar variación con signo
      let sign = value >= 0 ? '+' : '';
      return `${sign}${value.toFixed(1)}°C`;
    } else if (['fc', 'pasos', 'spo2'].includes(metric)) {
      // Valores enteros para ciertas métricas
      return Math.round(value);
    } else {
      // Una decimal para otras métricas
      return value.toFixed(1);
    }
  }
  
  // FUNCIÓN: Actualizar la tabla de eventos para cada métrica CON PAGINACIÓN
  function updateEventsTable(metric, events) {
    var tbody = document.getElementById(metric + '-events');
    var paginationContainer = document.getElementById(metric + '-pagination');
    if (!tbody) return;
    
    // Limpiar contenido anterior
    tbody.innerHTML = '';
    if (paginationContainer) paginationContainer.innerHTML = '';
    
    // Si no hay eventos, mostrar mensaje
    if (!events || events.length === 0) {
      var tr = document.createElement('tr');
      var td = document.createElement('td');
      td.colSpan = 4;
      td.className = 'text-center';
      td.textContent = 'No significant events detected';
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }
    
    // Configuración de paginación
    const eventsPerPage = 5;
    const totalPages = Math.ceil(events.length / eventsPerPage);
    let currentPage = 1;
    
    // Función para mostrar eventos de la página actual
    function displayEvents(page) {
      tbody.innerHTML = '';
      currentPage = page;
      
      const start = (page - 1) * eventsPerPage;
      const end = Math.min(start + eventsPerPage, events.length);
      const pageEvents = events.slice(start, end);
      
      pageEvents.forEach(event => {
        var tr = document.createElement('tr');
        
        // Columna de tiempo
        var tdTime = document.createElement('td');
        tdTime.textContent = event.timestamp || 'N/A';
        tr.appendChild(tdTime);
        
        // Columna de valor biométrico con formato mejorado
        var tdValue = document.createElement('td');
        var valueSpan = document.createElement('span');
        valueSpan.classList.add(event.is_abnormal ? 'text-warning' : 'text-success');
        
        // Formatear valor según tipo de métrica
        let formattedValue;
        if (metric === 'temperatura') {
          let sign = event.value >= 0 ? '+' : '';
          formattedValue = `${sign}${event.value.toFixed(1)}°C`;
        } else {
          formattedValue = `${event.value} ${window.metricConfig && window.metricConfig[metric]?.unit || ''}`;
        }
        
        // Añadir tooltip con razón de anormalidad
        if (event.is_abnormal) {
          let tooltipText = event.abnormal_reason || 'Valor anormal';
          valueSpan.innerHTML = `${formattedValue} 
                        <i class="fas fa-exclamation-triangle ml-1" data-toggle="tooltip" title="${tooltipText}"></i>`;
        } else {
          valueSpan.innerHTML = `${formattedValue} 
                        <i class="fas fa-check-circle ml-1" data-toggle="tooltip" title="Valor normal"></i>`;
        }
        
        tdValue.appendChild(valueSpan);
        tr.appendChild(tdValue);
        
        // Columna de actividad
        var tdActivity = document.createElement('td');
        tdActivity.innerHTML = `<strong>${event.activity_type || 'Unknown'}</strong>`;
        tr.appendChild(tdActivity);
        
        // Columna de detalles
        var tdDetails = document.createElement('td');
        if (event.details) {
          var detailsList = document.createElement('ul');
          detailsList.className = 'mb-0 pl-3';
          
          // Mostrar solo los 3 detalles más relevantes
          const keysToShow = Object.keys(event.details).slice(0, 3);
          
          keysToShow.forEach(key => {
            const value = event.details[key];
            if (value) {
              var li = document.createElement('li');
              li.innerHTML = `<small>${key}: <span class="text-primary">${value}</span></small>`;
              detailsList.appendChild(li);
            }
          });
          
          if (detailsList.children.length > 0) {
            tdDetails.appendChild(detailsList);
          } else {
            tdDetails.textContent = 'No additional details';
          }
        } else {
          tdDetails.textContent = 'No additional details';
        }
        tr.appendChild(tdDetails);
        
        tbody.appendChild(tr);
      });
      
      // Reinicializar tooltips para los nuevos elementos
      $('[data-toggle="tooltip"]').tooltip();
      
      // Actualizar controles de paginación
      updatePaginationControls();
    }
    
    // Función para actualizar controles de paginación
    function updatePaginationControls() {
      if (!paginationContainer) return;
      
      // No mostrar paginación si solo hay una página
      if (totalPages <= 1) {
        paginationContainer.innerHTML = '';
        
        // Si hay eventos pero no suficientes para paginar, mostrar un contador simple
        if (events.length > 0) {
          const countDiv = document.createElement('div');
          countDiv.className = 'text-center text-muted mt-2';
          countDiv.innerHTML = `<small>${events.length} evento${events.length !== 1 ? 's' : ''} en total</small>`;
          paginationContainer.appendChild(countDiv);
        }
        return;
      }
      
      paginationContainer.innerHTML = '';
      
      // Contenedor principal para centrar toda la paginación
      const paginationWrapper = document.createElement('div');
      paginationWrapper.className = 'd-flex flex-column align-items-center';
      
      // Contenedor de paginación con estilo
      const paginationNav = document.createElement('nav');
      paginationNav.setAttribute('aria-label', 'Events pagination');
      paginationNav.className = 'mb-2'; // Agregar margen inferior
      
      const paginationUl = document.createElement('ul');
      paginationUl.className = 'pagination pagination-sm justify-content-center';
      
      // Botón Anterior
      const prevLi = document.createElement('li');
      prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
      
      const prevLink = document.createElement('a');
      prevLink.className = 'page-link';
      prevLink.href = '#';
      prevLink.textContent = 'Prev';
      prevLink.addEventListener('click', function(e) {
        e.preventDefault();
        if (currentPage > 1) displayEvents(currentPage - 1);
      });
      
      prevLi.appendChild(prevLink);
      paginationUl.appendChild(prevLi);
      
      // Botones de páginas (con simplificación para muchas páginas)
      const maxPageButtons = 5;
      let startPage = Math.max(1, currentPage - Math.floor(maxPageButtons / 2));
      let endPage = Math.min(totalPages, startPage + maxPageButtons - 1);
      
      if (endPage - startPage + 1 < maxPageButtons) {
        startPage = Math.max(1, endPage - maxPageButtons + 1);
      }
      
      // Añadir primera página y elipsis si es necesario
      if (startPage > 1) {
        const firstLi = document.createElement('li');
        firstLi.className = 'page-item';
        
        const firstLink = document.createElement('a');
        firstLink.className = 'page-link';
        firstLink.href = '#';
        firstLink.textContent = '1';
        firstLink.addEventListener('click', function(e) {
          e.preventDefault();
          displayEvents(1);
        });
        
        firstLi.appendChild(firstLink);
        paginationUl.appendChild(firstLi);
        
        if (startPage > 2) {
          const ellipsisLi = document.createElement('li');
          ellipsisLi.className = 'page-item disabled';
          
          const ellipsisSpan = document.createElement('span');
          ellipsisSpan.className = 'page-link';
          ellipsisSpan.innerHTML = '&hellip;';
          
          ellipsisLi.appendChild(ellipsisSpan);
          paginationUl.appendChild(ellipsisLi);
        }
      }
      
      // Páginas numeradas
      for (let i = startPage; i <= endPage; i++) {
        const pageLi = document.createElement('li');
        pageLi.className = `page-item ${i === currentPage ? 'active' : ''}`;
        
        const pageLink = document.createElement('a');
        pageLink.className = 'page-link';
        pageLink.href = '#';
        pageLink.textContent = i;
        pageLink.addEventListener('click', function(e) {
          e.preventDefault();
          displayEvents(i);
        });
        
        pageLi.appendChild(pageLink);
        paginationUl.appendChild(pageLi);
      }
      
      // Añadir última página y elipsis si es necesario
      if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
          const ellipsisLi = document.createElement('li');
          ellipsisLi.className = 'page-item disabled';
          
          const ellipsisSpan = document.createElement('span');
          ellipsisSpan.className = 'page-link';
          ellipsisSpan.innerHTML = '&hellip;';
          
          ellipsisLi.appendChild(ellipsisSpan);
          paginationUl.appendChild(ellipsisLi);
        }
        
        const lastLi = document.createElement('li');
        lastLi.className = 'page-item';
        
        const lastLink = document.createElement('a');
        lastLink.className = 'page-link';
        lastLink.href = '#';
        lastLink.textContent = totalPages;
        lastLink.addEventListener('click', function(e) {
          e.preventDefault();
          displayEvents(totalPages);
        });
        
        lastLi.appendChild(lastLink);
        paginationUl.appendChild(lastLi);
      }
      
      // Botón Siguiente
      const nextLi = document.createElement('li');
      nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
      
      const nextLink = document.createElement('a');
      nextLink.className = 'page-link';
      nextLink.href = '#';
      nextLink.textContent = 'Next';
      nextLink.addEventListener('click', function(e) {
        e.preventDefault();
        if (currentPage < totalPages) displayEvents(currentPage + 1);
      });
      
      nextLi.appendChild(nextLink);
      paginationUl.appendChild(nextLi);
      
      paginationNav.appendChild(paginationUl);
      paginationWrapper.appendChild(paginationNav);
      
      // Contador de eventos debajo de la paginación
      const countDiv = document.createElement('div');
      countDiv.className = 'text-center text-muted';
      countDiv.innerHTML = `<small>${events.length} evento${events.length !== 1 ? 's' : ''} en total</small>`;
      paginationWrapper.appendChild(countDiv);
      
      paginationContainer.appendChild(paginationWrapper);
    }
    
    // Mostrar la primera página de eventos
    displayEvents(1);
  }
  
  // Función para formatear las etiquetas de tiempo de forma más legible
  function formatTimeLabel(timeStr) {
    if (!timeStr) return '';
    
    // Intentar diferentes formatos de tiempo
    try {
      // Si es un timestamp completo ISO
      if (timeStr.includes('T')) {
        const date = new Date(timeStr);
        if (!isNaN(date.getTime())) {
          return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
      }
      
      // Si es un formato timestamp estándar con espacio
      if (timeStr.includes(' ') && timeStr.length > 8) {
        const timePart = timeStr.split(' ')[1];
        if (timePart && timePart.includes(':')) {
          return timePart.substring(0, 5); // HH:MM
        }
      }
      
      // Si ya es una hora, solo devolver los primeros 5 caracteres (HH:MM)
      if (timeStr.includes(':') && timeStr.length >= 5) {
        return timeStr.substring(0, 5);
      }
    } catch (e) {
      console.warn("Error al formatear etiqueta de tiempo:", e);
    }
    
    return timeStr; // Devolver el original si no se puede formatear
  }
  
  // NUEVA FUNCIÓN: Extraer timestamps por hora para el eje X
  function extractHourlyTimestamps(timestamps, totalDuration) {
    if (!timestamps || timestamps.length === 0) return [];

    try {
      // Intentar convertir a objetos Date
      const dateTimes = timestamps
        .map(ts => {
          try {
            // Manejar diferentes formatos
            if (typeof ts === 'string') {
              if (ts.includes('T')) return new Date(ts); // ISO format
              if (ts.includes(' ') && ts.includes(':')) {
                const parts = ts.split(' ');
                return new Date(parts[0] + 'T' + parts[1]);
              }
            }
            return null;
          } catch (e) {
            return null;
          }
        })
        .filter(dt => dt !== null);

      if (dateTimes.length === 0) return [];

      // Ordenar los tiempos
      dateTimes.sort((a, b) => a - b);

      // Obtener la hora inicial y final
      const startTime = dateTimes[0];
      const endTime = dateTimes[dateTimes.length - 1];
      
      // Duración total en horas
      const durationHours = (endTime - startTime) / (1000 * 60 * 60);
      
      // Si la duración es menor a 2 horas, usamos intervalos más cortos
      const hourInterval = durationHours < 2 ? 0.5 : 1;
      
      // Generar timestamps por hora
      const hourlyTimestamps = [];
      let currentTime = new Date(startTime);
      
      // Añadir la hora inicial
      hourlyTimestamps.push({
        timestamp: currentTime.toISOString(),
        label: formatTimeLabel(currentTime.toTimeString()),
        position: 0 // Posición relativa (0 = inicio)
      });
      
      // Añadir horas intermedias
      while (currentTime < endTime) {
        currentTime = new Date(currentTime.getTime() + hourInterval * 60 * 60 * 1000);
        if (currentTime <= endTime) {
          const position = (currentTime - startTime) / (endTime - startTime);
          hourlyTimestamps.push({
            timestamp: currentTime.toISOString(),
            label: formatTimeLabel(currentTime.toTimeString()),
            position: position
          });
        }
      }
      
      return hourlyTimestamps;
    } catch (e) {
      console.error("Error al extraer timestamps por hora:", e);
      return [];
    }
  }

  // NUEVA FUNCIÓN: Optimizar la visualización de datos para mostrar una vista completa
  function createOptimizedView(originalData, originalLabels, eventIndices) {
    // Si hay pocos datos, mostrarlos todos
    if (originalData.length <= 1000) {
      return {
        data: originalData,
        indices: Array.from({ length: originalData.length }, (_, i) => i),
        labels: originalLabels
      };
    }

    // Crear arrays para datos y sus índices correspondientes
    const viewData = [];
    const viewIndices = [];
    const viewLabels = [];
    
    // Obtener los timestamps horarios para mejorar la visualización
    const hourlyTimestamps = extractHourlyTimestamps(originalLabels, originalData.length);
    
    // Convertir índices de eventos en un conjunto para búsqueda rápida
    const eventSet = new Set(eventIndices);
    
    // Incluir siempre el primer y último punto
    viewData.push(originalData[0]);
    viewIndices.push(0);
    viewLabels.push(originalLabels[0]);
    
    // Añadir todos los puntos de eventos importantes
    for (const idx of eventIndices) {
      if (idx > 0 && idx < originalData.length - 1) {  // Evitar duplicar primer/último punto
        viewData.push(originalData[idx]);
        viewIndices.push(idx);
        viewLabels.push(originalLabels[idx]);
      }
    }
    
    // Añadir puntos en intervalos regulares
    const step = Math.max(1, Math.floor(originalData.length / 100));
    for (let i = step; i < originalData.length - 1; i += step) {
      // Evitar duplicar puntos de eventos
      if (!eventSet.has(i)) {
        viewData.push(originalData[i]);
        viewIndices.push(i);
        viewLabels.push(originalLabels[i]);
      }
    }
    
    // Siempre incluir el último punto si no está ya incluido
    if (!eventSet.has(originalData.length - 1)) {
      viewData.push(originalData[originalData.length - 1]);
      viewIndices.push(originalData.length - 1);
      viewLabels.push(originalLabels[originalData.length - 1]);
    }
    
    // Ordenar por índice para mantener el orden cronológico
    const sortedItems = viewIndices.map((idx, pos) => ({ 
      idx, 
      data: viewData[pos],
      label: viewLabels[pos]
    })).sort((a, b) => a.idx - b.idx);
    
    return {
      data: sortedItems.map(item => item.data),
      indices: sortedItems.map(item => item.idx),
      labels: sortedItems.map(item => item.label)
    };
  }
  
  // Agregar banner de instrucciones de zoom
  function addZoomInstructionsBanner() {
    // Verificar si ya existe
    if (document.getElementById('zoom-instructions-banner')) return;
    
    // Encontrar el primer contenedor donde insertarlo
    const container = document.querySelector('.container-fluid');
    if (!container) return;
    
    const banner = document.createElement('div');
    banner.id = 'zoom-instructions-banner';
    banner.className = 'alert alert-info alert-dismissible fade show mb-4';
    banner.role = 'alert';
    banner.innerHTML = `
      <div class="d-flex align-items-center">
        <i class="fas fa-mouse mr-2"></i>
        <div><strong>Consejo:</strong> Puedes hacer zoom en los gráficos girando la rueda del ratón sobre ellos</div>
      </div>
      <button type="button" class="close" data-dismiss="alert" aria-label="Close">
        <span aria-hidden="true">&times;</span>
      </button>
    `;
    
    // Insertar al inicio del contenedor
    container.insertBefore(banner, container.firstChild);
  }
  
  // Agregar el banner de instrucciones solo una vez
  addZoomInstructionsBanner();
  
  // Función para añadir mensaje de zoom y botón de reinicio debajo del gráfico
  function addZoomMessage(chartContainer, chartInstance) {
    // Verificar si ya existe
    if (chartContainer.querySelector('.zoom-message-container')) return;
    
    // Crear el contenedor para el mensaje de zoom
    const zoomMessageContainer = document.createElement('div');
    zoomMessageContainer.className = 'zoom-message-container text-center mt-2';
    
    // Crear el mensaje
    const zoomMessage = document.createElement('div');
    zoomMessage.className = 'zoom-message text-muted small';
    zoomMessage.innerHTML = '<i class="fas fa-mouse mr-1"></i> Usa la rueda del ratón para hacer zoom';
    
    // Crear el botón de reinicio
    const resetButton = document.createElement('button');
    resetButton.type = 'button';
    resetButton.className = 'btn btn-sm btn-outline-primary ml-2';
    resetButton.innerHTML = '<i class="fas fa-undo mr-1"></i> Reiniciar zoom';
    resetButton.onclick = function() {
      // Intentar diferentes métodos para resetear el zoom
      try {
        if (chartInstance.resetZoom) {
          chartInstance.resetZoom();
        } else if (chartInstance.scales && chartInstance.scales.x) {
          // Alternativa para Chart.js más reciente
          chartInstance.scales.x.options.min = undefined;
          chartInstance.scales.x.options.max = undefined;
          chartInstance.scales.y.options.min = undefined;
          chartInstance.scales.y.options.max = undefined;
          chartInstance.update();
        } else {
          // Última opción: recrear el gráfico
          chartInstance.update();
        }
      } catch (error) {
        console.error("Error al reiniciar zoom:", error);
      }
    };
    
    // Añadir mensaje y botón al contenedor
    const wrapper = document.createElement('div');
    wrapper.className = 'd-flex align-items-center justify-content-center';
    wrapper.appendChild(zoomMessage);
    wrapper.appendChild(resetButton);
    
    zoomMessageContainer.appendChild(wrapper);
    
    // Insertar el contenedor después del canvas
    chartContainer.appendChild(zoomMessageContainer);
  }
  
  // Crear gráficos para cada dataset con mejoras visuales
  chartDatasets.forEach(dataset => {
    var metric = dataset.metric;
    if (!metric) return;
    
    var ctx = document.getElementById(metric + '-chart');
    if (!ctx) return;

    // Procesar eventos para esta métrica
    const events = eventsData[metric] || [];
    
    // Extraer índices de eventos importantes
    const eventIndices = events.map(event => event.index || 0);
    
    // Actualizar la tabla de eventos
    updateEventsTable(metric, events);
    
    // SOLUCIÓN MEJORADA: Preprocesar las etiquetas para el eje X
    const processedLabels = Array.isArray(chartLabels) ? chartLabels.map(label => {
      if (typeof label === 'string') {
        return formatTimeLabel(label);
      }
      return label;
    }) : [];
    
    // Crear vista optimizada para la visualización
    const optimizedView = createOptimizedView(
      dataset.data, 
      processedLabels.length === dataset.data.length ? processedLabels : Array(dataset.data.length).fill(''),
      eventIndices
    );
    
    const optimizedData = optimizedView.data;
    const optimizedIndices = optimizedView.indices;
    const optimizedLabels = optimizedView.labels;
    
    // Obtener el tipo de gráfico configurado
    var chartType = dataset.chart_type || defaultChartType || 'line';
    
    // Configuración para puntos en la gráfica
    var pointRadius = Array(optimizedData.length).fill(2);
    var pointBackgroundColor = Array(optimizedData.length).fill(dataset.borderColor);
    var pointBorderColor = Array(optimizedData.length).fill('#fff');
    var pointBorderWidth = Array(optimizedData.length).fill(1);
    
    // Resaltar puntos de eventos importantes
    events.forEach(event => {
      const idx = optimizedIndices.indexOf(event.index);
      if (idx !== -1) {
        pointRadius[idx] = event.is_abnormal ? 6 : 5;
        pointBackgroundColor[idx] = event.is_abnormal ? '#fb6340' : '#2dce89';
        pointBorderColor[idx] = '#ffffff';
        pointBorderWidth[idx] = 2;
      }
    });
    
    // Información para tooltips (contexto de cada punto)
    var activityInfo = Array(optimizedData.length).fill(null);
    
    // Asociar información de eventos con sus puntos correspondientes
    events.forEach(event => {
      const idx = optimizedIndices.indexOf(event.index);
      if (idx !== -1) {
        activityInfo[idx] = {
          timestamp: event.timestamp || '',
          activity_type: event.activity_type || 'Actividad',
          is_abnormal: event.is_abnormal || false,
          abnormal_reason: event.abnormal_reason || 'Normal',
          details: event.details || {},
          isEvent: true
        };
      }
    });
    
    // Completar información de puntos que no tienen eventos asociados
    for (let i = 0; i < optimizedData.length; i++) {
      if (!activityInfo[i]) {
        activityInfo[i] = {
          timestamp: optimizedLabels[i] || '',
          activity_type: '',  // Eliminado "Punto de datos"
          is_abnormal: false,
          abnormal_reason: 'Valor normal',
          details: {},
          isEvent: false
        };
      }
    }
    
    // MEJORA VISUAL: Extraer horas para anotaciones y líneas de referencia
    const hourlyTimestamps = extractHourlyTimestamps(chartLabels, dataset.data.length);
    
    // MEJORA VISUAL: Preparar anotaciones y líneas de referencia
    const annotations = {};
    
    // Añadir líneas verticales para cada hora
    hourlyTimestamps.forEach((hourPoint, idx) => {
      // Encontrar el índice más cercano en datos optimizados
      const nearestIdx = optimizedIndices.reduce((prev, curr, i) => {
        const prevDiff = Math.abs((prev / dataset.data.length) - hourPoint.position);
        const currDiff = Math.abs((curr / dataset.data.length) - hourPoint.position);
        return currDiff < prevDiff ? curr : prev;
      }, optimizedIndices[0]);
      
      const annotationIndex = optimizedIndices.indexOf(nearestIdx);
      if (annotationIndex !== -1) {
        annotations[`hour-line-${idx}`] = {
          type: 'line',
          scaleID: 'x',
          value: annotationIndex,
          borderColor: 'rgba(136, 152, 170, 0.3)',
          borderWidth: 1,
          borderDash: [5, 5],
          label: {
            content: hourPoint.label,
            enabled: true,
            position: 'start',
            backgroundColor: 'rgba(136, 152, 170, 0.7)',
            color: '#ffffff',
            font: {
              size: 10
            }
          }
        };
      }
    });
    
    // MEJORA: Añadir líneas horizontales para valores de referencia según métrica
    if (metric === 'fc') {
      // Línea para frecuencia cardíaca normal máxima en reposo
      annotations['fc-normal-max'] = {
        type: 'line',
        scaleID: 'y',
        value: 100,
        borderColor: 'rgba(245, 54, 92, 0.5)',
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          content: 'FC normal máx.',
          enabled: true,
          position: 'end',
          backgroundColor: 'rgba(245, 54, 92, 0.7)',
          color: '#ffffff',
          font: { size: 10 }
        }
      };
    } else if (metric === 'spo2') {
      // Línea para nivel óptimo de saturación de oxígeno
      annotations['spo2-normal'] = {
        type: 'line',
        scaleID: 'y',
        value: 95,
        borderColor: 'rgba(29, 140, 248, 0.5)',
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          content: 'SpO₂ óptimo',
          enabled: true,
          position: 'end',
          backgroundColor: 'rgba(29, 140, 248, 0.7)',
          color: '#ffffff',
          font: { size: 10 }
        }
      };
    }
    
    // MEJORA: Configuración del gráfico con mejor visualización
    var config = {
      type: chartType === 'area' ? 'line' : chartType,
      data: {
        labels: optimizedLabels,
        datasets: [{
          label: metricConfig[metric]?.displayName || metric,
          data: optimizedData,
          borderColor: dataset.borderColor,
          backgroundColor: dataset.backgroundColor,
          fill: chartType === 'area' ? 'origin' : false,
          tension: chartType === 'area' || chartType === 'line' ? 0.3 : 0,
          pointRadius: chartType === 'bar' ? 0 : pointRadius,
          pointBackgroundColor: pointBackgroundColor,
          pointBorderColor: pointBorderColor,
          pointBorderWidth: pointBorderWidth,
          pointHoverRadius: chartType === 'bar' ? 0 : 8,
          pointHoverBackgroundColor: dataset.borderColor,
          pointHoverBorderColor: '#fff',
          borderWidth: chartType === 'bar' ? 1 : 2,
          barPercentage: chartType === 'bar' ? 0.8 : 1,
          categoryPercentage: chartType === 'bar' ? 0.9 : 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          tooltip: {
            enabled: true,
            position: 'nearest',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleFont: { size: 13 },
            bodyFont: { size: 12 },
            padding: 10,
            displayColors: false,
            callbacks: {
              title: function(context) {
                const dataIndex = context[0].dataIndex;
                const info = activityInfo[dataIndex];
                
                // Mostrar solo la hora como título
                if (info && info.timestamp) {
                  return info.timestamp;
                }
                return optimizedLabels[dataIndex] || 'Tiempo no disponible';
              },
              label: function(context) {
                const dataIndex = context.dataIndex;
                let lines = [];
                
                // Línea 1: Valor básico con unidades
                var basicLabel = context.dataset.label || '';
                if (basicLabel) basicLabel += ': ';
                
                if (context.parsed.y !== null) {
                  // Formateo especial para temperatura
                  if (metric === 'temperatura') {
                    let value = context.parsed.y;
                    let sign = value >= 0 ? '+' : '';
                    basicLabel += `${sign}${value.toFixed(1)}°C`;
                  } else {
                    basicLabel += formatMetricValue(context.parsed.y, metric);
                    if (metricConfig[metric]?.unit) {
                      basicLabel += ' ' + metricConfig[metric].unit;
                    }
                  }
                }
                
                lines.push(basicLabel);
                
                // Mostrar información relevante
                const info = activityInfo[dataIndex];
                
                if (info) {
                  // Solo mostrar actividad si es un evento real con actividad definida
                  if (info.isEvent && info.activity_type) {
                    lines.push(`Actividad: ${info.activity_type}`);
                  }
                  
                  // Estado (siempre mostrar, con razón si es anormal)
                  if (info.is_abnormal) {
                    lines.push(`Estado: Anormal (${info.abnormal_reason || 'Sin detalles'})`);
                  } else {
                    lines.push('Estado: Normal');
                  }
                  
                  // Detalles de la aplicación y acción (máximo 2 detalles)
                  if (info.details) {
                    // Priorizar mostrar la aplicación
                    if (info.details['app']) {
                      lines.push(`App: ${info.details['app']}`);
                    }
                    
                    // Buscar información de acción relevante
                    const actionKeys = ['action', 'Element Text', 'Element Type', 'Input'];
                    for (const key of actionKeys) {
                      if (info.details[key]) {
                        lines.push(`${key === 'action' ? 'Acción' : key}: ${info.details[key]}`);
                        break; // Solo mostrar una acción
                      }
                    }
                  }
                }
                
                return lines;
              }
            }
          },
          legend: {
            display: false
          },
          annotation: {
            annotations: annotations
          },
          zoom: {
            pan: {
              enabled: true,
              mode: 'x'
            },
            zoom: {
              wheel: { 
                enabled: true,
                speed: 0.1,
                modifierKey: null  // No requiere tecla modificadora (Ctrl)
              },
              pinch: { 
                enabled: true 
              },
              mode: 'x',
              onZoom: function() {
                // Este callback es importante para que el evento de zoom se registre correctamente
              }
            },
            limits: {
              x: {min: 'original', max: 'original'},
              y: {min: 'original', max: 'original'}
            }
          }
        },
        scales: {
          x: {
            grid: { 
              display: true, 
              color: "rgba(0,0,0,0.05)",
              drawBorder: true
            },
            ticks: {
              maxRotation: 45, // Rotar las etiquetas para evitar superposición
              minRotation: 45, // Mantener una rotación constante
              autoSkip: true, // Activar el salto automático de etiquetas
              autoSkipPadding: 15, // Espacio mínimo entre etiquetas
              callback: function(val, index) {
                // Mostrar etiquetas en intervalos para evitar sobrecargar
                if (optimizedLabels.length > 50) {
                  // Para muchos datos, mostrar menos etiquetas
                  if (index % 5 === 0) {
                    return this.getLabelForValue(val);
                  }
                  return '';
                }
                return this.getLabelForValue(val);
              },
              color: 'rgba(0, 0, 0, 0.75)', // Hacer las etiquetas más visibles
              font: {
                size: 10,
                weight: 'bold' // Fuente más destacada
              }
            }
          },
          y: {
            grid: { 
              display: true, 
              color: "rgba(0,0,0,0.05)",
              drawBorder: true 
            },
            beginAtZero: metric !== 'fc' && metric !== 'spo2',
            ticks: {
              padding: 8,
              callback: function(value) {
                if (metric === 'temperatura') {
                  return (value >= 0 ? '+' : '') + value.toFixed(1) + '°C';
                }
                return value;
              }
            }
          }
        }
      }
    };
    
    // Ajustes específicos para gráficos de área
    if (chartType === 'area') {
      // Aumentar la opacidad del color de fondo para área
      if (config.data.datasets[0].backgroundColor.includes('rgba')) {
        config.data.datasets[0].backgroundColor = config.data.datasets[0].backgroundColor.replace(
          /rgba\((\d+),\s*(\d+),\s*(\d+),\s*[\d.]+\)/,
          'rgba($1, $2, $3, 0.5)'
        );
      } else {
        // Si no es rgba, añadir opacidad
        var color = config.data.datasets[0].backgroundColor;
        if (color.startsWith('#')) {
          // Convertir HEX a RGBA
          var r = parseInt(color.slice(1, 3), 16);
          var g = parseInt(color.slice(3, 5), 16);
          var b = parseInt(color.slice(5, 7), 16);
          config.data.datasets[0].backgroundColor = `rgba(${r}, ${g}, ${b}, 0.5)`;
        }
      }
    }
    
    // Optimizaciones específicas según métrica y tipo de gráfico
    if (metric === 'fc' && chartType === 'bar') {
      config.data.datasets[0].backgroundColor = 'rgba(245, 54, 92, 0.6)';
    }
    
    if (metric === 'pasos' && chartType === 'line') {
      config.data.datasets[0].stepped = true;
    }
    
    // Ajustes específicos para temperatura
    if (metric === 'temperatura') {
      try {
        // Colores especiales para temperaturas positivas y negativas
        const gradientAboveZero = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
        gradientAboveZero.addColorStop(0, 'rgba(251, 99, 64, 0.8)');
        gradientAboveZero.addColorStop(1, 'rgba(251, 99, 64, 0.1)');
        
        const gradientBelowZero = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
        gradientBelowZero.addColorStop(0, 'rgba(94, 114, 228, 0.1)');
        gradientBelowZero.addColorStop(1, 'rgba(94, 114, 228, 0.8)');
        
        // Aplicar colores según el tipo de gráfico
        if (chartType === 'area') {
          // Para área, usar gradientes según el valor
          config.data.datasets[0].backgroundColor = function(context) {
            const value = context.raw;
            return value >= 0 ? gradientAboveZero : gradientBelowZero;
          };
        } else if (chartType === 'bar') {
          // Para barras, color según el valor
          config.data.datasets[0].backgroundColor = function(context) {
            const value = context.raw;
            return value >= 0 ? 'rgba(251, 99, 64, 0.6)' : 'rgba(94, 114, 228, 0.6)';
          };
        }
      } catch (error) {
        console.warn("Error configurando gradientes para temperatura:", error);
      }
    }
    
    // MEJORA: Cargar plugins necesarios de forma dinámica y luego crear el gráfico
    function loadPlugins() {
      return new Promise((resolve, reject) => {
        // Lista de plugins que necesitamos
        const plugins = [
          {
            name: 'chartjs-plugin-annotation',
            url: 'https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.1.0/dist/chartjs-plugin-annotation.min.js'
          },
          {
            name: 'chartjs-plugin-zoom',
            url: 'https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.0/dist/chartjs-plugin-zoom.min.js',
            requires: ['hammer']
          },
          {
            name: 'hammer',
            url: 'https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js'
          }
        ];
        
        // Registrar los plugins que ya están cargados
        const loadedPlugins = {};
        plugins.forEach(plugin => {
          loadedPlugins[plugin.name] = window[plugin.name] !== undefined;
        });
        
        // Función para cargar un plugin
        function loadPlugin(plugin) {
          return new Promise((resolvePlugin, rejectPlugin) => {
            // Si ya está cargado, resolver inmediatamente
            if (loadedPlugins[plugin.name]) {
              resolvePlugin();
              return;
            }
            
            // Verificar dependencias
            if (plugin.requires) {
              for (const dep of plugin.requires) {
                if (!loadedPlugins[dep]) {
                  rejectPlugin(`El plugin ${plugin.name} requiere ${dep} que no está cargado`);
                  return;
                }
              }
            }
            
            // Cargar el script
            const script = document.createElement('script');
            script.src = plugin.url;
            script.onload = () => {
              loadedPlugins[plugin.name] = true;
              resolvePlugin();
            };
            script.onerror = () => rejectPlugin(`Error al cargar ${plugin.name}`);
            document.head.appendChild(script);
          });
        }
        
        // Cargar plugins en orden específico (primero hammer, luego los demás)
        loadPlugin(plugins.find(p => p.name === 'hammer'))
          .then(() => {
            // Cargar los plugins restantes en paralelo
            return Promise.all(
              plugins
                .filter(p => p.name !== 'hammer')
                .map(loadPlugin)
            );
          })
          .then(resolve)
          .catch(reject);
      });
    }
    
    // Crear el gráfico con los plugins necesarios
    loadPlugins().then(() => {
      // Registrar los plugins globalmente para Chart.js si es necesario
      if (window['chartjs-plugin-zoom'] && typeof Chart.register === 'function') {
        Chart.register(window['chartjs-plugin-zoom']);
      }
      
      if (window['chartjs-plugin-annotation'] && typeof Chart.register === 'function') {
        Chart.register(window['chartjs-plugin-annotation']);
      }
      
      // Crear el gráfico una vez que todos los plugins estén cargados
      charts[metric] = new Chart(ctx, config);
      
      // Encontrar el contenedor del gráfico para añadir el mensaje de zoom
      const chartContainer = ctx.parentNode;
      addZoomMessage(chartContainer, charts[metric]);
      
    }).catch(err => {
      console.error('Error al cargar plugins:', err);
      
      // Si falla la carga de plugins, crear una versión simplificada del gráfico
      delete config.options.plugins.annotation;
      delete config.options.plugins.zoom;
      
      charts[metric] = new Chart(ctx, config);
    });
  });
  
  // Navegación de pestañas usando JavaScript puro
  var tabLinks = document.querySelectorAll('.custom-tabs .nav-link');
  var tabContents = document.querySelectorAll('.tab-pane');
  
  // Función para mostrar una pestaña específica
  function showTab(tabId) {
    // Ocultar todos los contenidos de pestañas
    tabContents.forEach(function(content) {
      content.classList.remove('show', 'active');
    });
    
    // Desactivar todas las pestañas
    tabLinks.forEach(function(link) {
      link.classList.remove('active');
      link.setAttribute('aria-selected', 'false');
    });
    
    // Activar la pestaña seleccionada
    var selectedTab = document.getElementById(tabId + '-tab');
    if (selectedTab) {
      selectedTab.classList.add('active');
      selectedTab.setAttribute('aria-selected', 'true');
    }
    
    // Mostrar el contenido de la pestaña
    var selectedContent = document.getElementById(tabId + '-content');
    if (selectedContent) {
      selectedContent.classList.add('show', 'active');
    }
    
    // Redimensionar el gráfico si existe
    if (charts[tabId]) {
      setTimeout(function() {
        charts[tabId].resize();
      }, 50);
    }
  }
  
  // Asignar eventos de clic a las pestañas
  tabLinks.forEach(function(link) {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      var tabId = this.id.replace('-tab', '');
      showTab(tabId);
    });
  });
  
  // Activar la primera pestaña por defecto
  if (tabLinks.length > 0) {
    var firstTabId = tabLinks[0].id.replace('-tab', '');
    showTab(firstTabId);
  }
  
  // Aplicar colores a los indicadores
  for (const [key, indicator] of Object.entries(indicatorsData)) {
    const card = document.getElementById('indicator-' + key);
    if (card && indicator.color) {
      // Usar las clases predefinidas
      if (indicator.color === 'primary') card.classList.add('border-primary');
      else if (indicator.color === 'danger') card.classList.add('border-danger');
      else if (indicator.color === 'warning') card.classList.add('border-warning');
      else if (indicator.color === 'success') card.classList.add('border-success');
      else if (indicator.color === 'info') card.classList.add('border-info');
      else if (indicator.color === 'purple') card.classList.add('border-purple');
      else card.style.borderColor = indicator.color; // Fallback a estilo inline
    }
  }
  
  // Eliminar cualquier scrollbar innecesario
  document.querySelectorAll('.card-body, .nav-tabs-container, .custom-tabs, .tab-content').forEach(function(element) {
    element.classList.add('no-scroll');
  });
  
  // Arreglo de dropdowns
  function fixDropdowns() {
    document.querySelectorAll('.dropdown-fix').forEach(el => el.remove());
    document.querySelectorAll('.dropdown-toggle').forEach(toggle => {
      toggle.removeEventListener('click', handleDropdownToggle);
      toggle.addEventListener('click', handleDropdownToggle);
    });
  }
  
  function handleDropdownToggle(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const toggle = this;
    const dropdownMenu = toggle.nextElementSibling;
    
    if (!dropdownMenu || !dropdownMenu.classList.contains('dropdown-menu')) return;
    
    const existingFixedDropdown = document.querySelector(`.dropdown-fix[data-original-id="${dropdownMenu.id}"]`);
    if (existingFixedDropdown) {
      existingFixedDropdown.remove();
      return;
    }
    
    document.querySelectorAll('.dropdown-fix').forEach(el => el.remove());
    
    const clonedDropdown = dropdownMenu.cloneNode(true);
    const rect = toggle.getBoundingClientRect();
    
    const isNotification = dropdownMenu.classList.contains('dropdown-menu-xl') || 
                        dropdownMenu.classList.contains('dropdown-menu-notification');
    const isUserMenu = toggle.querySelector('.ni-settings-gear-65') !== null;
    
    clonedDropdown.setAttribute('data-original-id', dropdownMenu.id || '');
    clonedDropdown.classList.add('dropdown-fix');
    
    if (isNotification) {
      clonedDropdown.style.right = (window.innerWidth - rect.right) + 'px';
      clonedDropdown.style.top = (rect.bottom + window.scrollY) + 'px';
      clonedDropdown.classList.add('dropdown-menu-notification');
    } else if (isUserMenu) {
      clonedDropdown.style.right = (window.innerWidth - rect.right) + 'px';
      clonedDropdown.style.top = (rect.bottom + window.scrollY) + 'px';
    } else {
      clonedDropdown.style.left = rect.left + 'px';
      clonedDropdown.style.top = (rect.bottom + window.scrollY) + 'px';
      clonedDropdown.style.minWidth = rect.width + 'px';
    }
    
    document.body.appendChild(clonedDropdown);
    
    setTimeout(() => {
      document.addEventListener('click', function closeDropdown(e) {
        if (!clonedDropdown.contains(e.target) && e.target !== toggle) {
          clonedDropdown.remove();
          document.removeEventListener('click', closeDropdown);
        }
      });
      
      clonedDropdown.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
          clonedDropdown.remove();
        });
      });
    }, 10);
  }
  
  fixDropdowns();
  setTimeout(fixDropdowns, 500);
  
  // Observar cambios en el DOM
  const observer = new MutationObserver(() => {
    setTimeout(fixDropdowns, 10);
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['class', 'style']
  });
  
  window.addEventListener('resize', fixDropdowns);
  window.addEventListener('scroll', fixDropdowns);
  
  // Inicializar tooltips
  $(function () {
    $('[data-toggle="tooltip"]').tooltip();
  });
});

document.getElementById('scenarioSelect').addEventListener('change', function() {
  const reportId = this.value;
  if (reportId) {
    window.location.href = `/wearabletracking/biometric-report/detail/${reportId}/`;
  }
});