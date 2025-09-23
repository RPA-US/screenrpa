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
  
  // Crear gráficos para cada dataset
  chartDatasets.forEach(dataset => {
    var metric = dataset.metric;
    if (!metric) return;
    
    var ctx = document.getElementById(metric + '-chart');
    if (!ctx) return;
    
    // SOLUCIÓN MEJORADA: Usar directamente los eventos del backend sin reprocesarlos
    // El backend ya se encarga de garantizar que todos los eventos importantes estén incluidos
    function processEvents(rawEvents, dataLength) {
      console.log(`Procesando ${rawEvents ? rawEvents.length : 0} eventos para ${metric}`);
      
      if (!rawEvents || !rawEvents.length) {
        return [];
      }

      // Los eventos ya vienen con el índice correcto del backend,
      // solo validamos que el índice esté dentro del rango de datos
      return rawEvents.filter(event => 
        event.index !== undefined && 
        event.index >= 0 && 
        event.index < dataLength
      );
    }
    
    // SOLUCIÓN: Procesar eventos para esta métrica
    const tableEvents = processEvents(eventsData[metric], dataset.data.length);
    
    // Actualizar la tabla con los eventos procesados
    updateEventsTable(metric, tableEvents);
    
    // Obtener el tipo de gráfico configurado
    var chartType = dataset.chart_type || defaultChartType || 'line';
    console.log(`Tipo de gráfico para ${metric}: ${chartType}`);
    
    // SOLUCIÓN: Configuración para puntos en la gráfica
    // 1. Configuración base para todos los puntos
    var pointRadius = Array(dataset.data.length).fill(2); // Puntos más pequeños por defecto
    var pointBackgroundColor = Array(dataset.data.length).fill(dataset.borderColor);
    var pointBorderColor = Array(dataset.data.length).fill('#fff');
    var pointBorderWidth = Array(dataset.data.length).fill(1);
    
    // 2. CLAVE: Resaltar SOLO los puntos de eventos (ahora funcionará porque el backend garantiza su inclusión)
    tableEvents.forEach(event => {
      if (event.index >= 0 && event.index < dataset.data.length) {
        pointRadius[event.index] = 6; // Puntos más grandes para eventos
        pointBackgroundColor[event.index] = event.is_abnormal ? '#fb6340' : '#2dce89'; // Rojo para anormales, verde para normales
        pointBorderColor[event.index] = '#ffffff';
        pointBorderWidth[event.index] = 2;
      }
    });
    
    // Información de actividad para todos los puntos (para tooltips)
    var activityInfo = Array(dataset.data.length);
    
    // PARTE 1: Asignar información de eventos a sus puntos correspondientes
    tableEvents.forEach(event => {
      if (event.index >= 0 && event.index < dataset.data.length) {
        activityInfo[event.index] = {
          timestamp: event.timestamp || chartLabels[event.index] || '',
          activity_type: event.activity_type || 'Actividad',
          is_abnormal: event.is_abnormal || false,
          abnormal_reason: event.abnormal_reason || 'Normal',
          details: event.details || {},
          isEvent: true
        };
      }
    });
    
    // PARTE 2: Asignar información aproximada a puntos sin eventos directos
    for (let i = 0; i < dataset.data.length; i++) {
      if (!activityInfo[i]) {
        // Buscar el evento más cercano para obtener información relacionada
        let nearestEvent = null;
        let minDistance = Infinity;
        
        tableEvents.forEach(event => {
          const distance = Math.abs(event.index - i);
          if (distance < minDistance) {
            minDistance = distance;
            nearestEvent = event;
          }
        });
        
        // Usar información del evento más cercano o información genérica
        if (nearestEvent) {
          activityInfo[i] = {
            timestamp: chartLabels[i] || '',
            activity_type: nearestEvent.activity_type || 'Actividad',
            is_abnormal: nearestEvent.is_abnormal || false,
            abnormal_reason: nearestEvent.abnormal_reason || 'Normal',
            details: nearestEvent.details || {},
            isEvent: false,
            nearestEventDistance: minDistance
          };
        } else {
          activityInfo[i] = {
            timestamp: chartLabels[i] || '',
            activity_type: 'Actividad regular',
            is_abnormal: false,
            abnormal_reason: 'Sin eventos significativos',
            details: {},
            isEvent: false
          };
        }
      }
    }
    
    // Configuración del gráfico
    var config = {
      type: chartType === 'area' ? 'line' : chartType,
      data: {
        labels: chartLabels,
        datasets: [{
          label: metricConfig[metric]?.displayName || metric,
          data: dataset.data,
          borderColor: dataset.borderColor,
          backgroundColor: dataset.backgroundColor,
          fill: chartType === 'area' ? 'origin' : false,
          tension: chartType === 'area' || chartType === 'line' ? 0.4 : 0,
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
        plugins: {
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              title: function(context) {
                return chartLabels[context[0].dataIndex];
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
                    basicLabel += context.parsed.y;
                    if (metricConfig[metric]?.unit) {
                      basicLabel += ' ' + metricConfig[metric].unit;
                    }
                  }
                }
                
                lines.push(basicLabel);
                
                // Mostrar información de actividad para todos los puntos
                const info = activityInfo[dataIndex];
                
                if (info) {
                  // Actividad
                  if (info.isEvent) {
                    lines.push(`Actividad: ${info.activity_type || 'Desconocida'} ✓`);
                  } else {
                    lines.push(`Actividad: ${info.activity_type || 'Desconocida'}`);
                  }
                  
                  // Estado
                  if (info.is_abnormal) {
                    lines.push(`Estado: Anormal (${info.abnormal_reason || 'Sin detalles'})`);
                  } else {
                    lines.push('Estado: Normal');
                  }
                  
                  // Detalles si existen
                  if (info.details) {
                    const keys = Object.keys(info.details);
                    for (let i = 0; i < Math.min(2, keys.length); i++) {
                      const key = keys[i];
                      if (info.details[key]) {
                        lines.push(`${key}: ${info.details[key]}`);
                      }
                    }
                  }
                  
                  // Indicar aproximación
                  if (!info.isEvent && info.nearestEventDistance) {
                    lines.push(`(Información aproximada)`);
                  }
                }
                
                return lines;
              }
            }
          },
          legend: {
            display: false
          }
        },
        hover: {
          mode: 'index',
          intersect: false
        },
        scales: {
          x: {
            grid: {
              display: true,
              color: "rgba(0,0,0,0.05)"
            }
          },
          y: {
            grid: {
              display: true,
              color: "rgba(0,0,0,0.05)"
            },
            beginAtZero: metric !== 'fc',
            ticks: metric === 'temperatura' ? {
              callback: function(value) {
                return (value >= 0 ? '+' : '') + value.toFixed(1) + '°C';
              }
            } : undefined
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
          'rgba($1, $2, $3, 0.5)' // Mayor opacidad (0.5) para que se vea mejor el área
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
      // Colores especiales para temperaturas positivas y negativas
      const gradientAboveZero = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
      gradientAboveZero.addColorStop(0, 'rgba(251, 99, 64, 0.8)');   // Rojo cálido arriba
      gradientAboveZero.addColorStop(1, 'rgba(251, 99, 64, 0.1)');   // Transparente abajo
      
      const gradientBelowZero = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
      gradientBelowZero.addColorStop(0, 'rgba(94, 114, 228, 0.1)');  // Transparente arriba
      gradientBelowZero.addColorStop(1, 'rgba(94, 114, 228, 0.8)');  // Azul frío abajo
      
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
    }
    
    // Para gráficos de barras con muchos datos, reducir la muestra si es necesario
    if (chartType === 'bar' && dataset.data.length > 30) {
      var decimation = Math.ceil(dataset.data.length / 30);
      var decimatedData = [];
      var decimatedLabels = [];
      
      for (var i = 0; i < dataset.data.length; i += decimation) {
        // Calculamos el promedio de este segmento
        var sum = 0;
        var count = 0;
        for (var j = i; j < i + decimation && j < dataset.data.length; j++) {
          if (dataset.data[j] !== null) {
            sum += dataset.data[j];
            count++;
          }
        }
        decimatedData.push(count > 0 ? sum / count : null);
        decimatedLabels.push(chartLabels[i]);
      }
      
      // Reemplazamos los datos en el config
      config.data.labels = decimatedLabels;
      config.data.datasets[0].data = decimatedData;
    }
    
    // Crear el gráfico
    charts[metric] = new Chart(ctx.getContext('2d'), config);
  });
  
  // SOLUCIÓN COMPLETA: Navegación de pestañas usando JavaScript puro
  // Esto evita dependencias con jQuery y soluciona los problemas de navegación
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
  (function removeUnnecessaryScrollbars() {
    // Eliminar scrollbars de todos los contenedores relevantes
    document.querySelectorAll('.card-body, .nav-tabs-container, .custom-tabs, .tab-content').forEach(function(element) {
      element.classList.add('no-scroll');
    });
    
    // Asegurar que no haya overflow visible que cause scrollbars
    document.querySelectorAll('.row').forEach(function(row) {
      row.style.marginRight = '0';
      row.style.marginLeft = '0';
    });
  })();
  
function fixDropdownsCompletely() {
    // Eliminar cualquier dropdown ya fijado previamente
    document.querySelectorAll('.dropdown-fix').forEach(el => el.remove());
    
    // Manejar los toggles de dropdown correctamente
    document.querySelectorAll('.dropdown-toggle').forEach(toggle => {
      // Eliminar cualquier manejador de eventos previo
      toggle.removeEventListener('click', handleDropdownToggle);
      toggle.addEventListener('click', handleDropdownToggle);
    });
  }
  
  function handleDropdownToggle(e) {
    e.preventDefault();
    e.stopPropagation();
    
    // Obtener el botón que activó el dropdown
    const toggle = this;
    // Encontrar el dropdown correspondiente
    const dropdownMenu = toggle.nextElementSibling;
    
    if (!dropdownMenu || !dropdownMenu.classList.contains('dropdown-menu')) return;
    
    // Si ya hay un dropdown fijo con el mismo ID, eliminarlo (toggle)
    const existingFixedDropdown = document.querySelector(`.dropdown-fix[data-original-id="${dropdownMenu.id}"]`);
    if (existingFixedDropdown) {
      existingFixedDropdown.remove();
      return;
    }
    
    // Eliminar cualquier otro dropdown fijado
    document.querySelectorAll('.dropdown-fix').forEach(el => el.remove());
    
    // Clonar el dropdown para posicionarlo absolutamente
    const clonedDropdown = dropdownMenu.cloneNode(true);
    const rect = toggle.getBoundingClientRect();
    
    // Determinar si es dropdown de notificaciones o perfil
    const isNotification = dropdownMenu.classList.contains('dropdown-menu-xl') || 
                          dropdownMenu.classList.contains('dropdown-menu-notification');
    const isUserMenu = toggle.querySelector('.ni-settings-gear-65') !== null;
    
    // Establecer ID de referencia para poder identificarlo después
    clonedDropdown.setAttribute('data-original-id', dropdownMenu.id || '');
    
    // Aplicar clases y estilos
    clonedDropdown.classList.add('dropdown-fix');
    
    // Posicionamiento específico según el tipo de dropdown
    if (isNotification) {
      // Dropdown de notificaciones - a la derecha
      clonedDropdown.style.right = (window.innerWidth - rect.right) + 'px';
      clonedDropdown.style.top = (rect.bottom + window.scrollY) + 'px';
      clonedDropdown.classList.add('dropdown-menu-notification');
    } else if (isUserMenu) {
      // Dropdown de usuario - a la derecha
      clonedDropdown.style.right = (window.innerWidth - rect.right) + 'px';
      clonedDropdown.style.top = (rect.bottom + window.scrollY) + 'px';
    } else {
      // Otros dropdowns - debajo del botón
      clonedDropdown.style.left = rect.left + 'px';
      clonedDropdown.style.top = (rect.bottom + window.scrollY) + 'px';
      clonedDropdown.style.minWidth = rect.width + 'px';
    }
    
    // Agregar al body para evitar problemas de contención
    document.body.appendChild(clonedDropdown);
    
    // Cerrar al hacer clic en cualquier lugar
    setTimeout(() => {
      document.addEventListener('click', function closeDropdown(e) {
        if (!clonedDropdown.contains(e.target) && e.target !== toggle) {
          clonedDropdown.remove();
          document.removeEventListener('click', closeDropdown);
        }
      });
      
      // También cerrar al hacer clic en elementos internos del dropdown
      clonedDropdown.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
          clonedDropdown.remove();
        });
      });
    }, 10);
  }
  
  // Ejecutar la solución inmediatamente
  fixDropdownsCompletely();
  
  // Y también después de un tiempo para asegurar que todo esté cargado
  setTimeout(fixDropdownsCompletely, 500);
  
  // Observar cambios en el DOM para mantener la funcionalidad
  const observer = new MutationObserver(() => {
    setTimeout(fixDropdownsCompletely, 10);
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['class', 'style']
  });
  
  // Asegurarse de que funcione incluso después de cambios en la ventana
  window.addEventListener('resize', fixDropdownsCompletely);
  window.addEventListener('scroll', fixDropdownsCompletely);
  
  // Inicializar todos los tooltips en el documento
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