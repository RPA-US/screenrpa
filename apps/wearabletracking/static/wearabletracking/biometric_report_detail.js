document.addEventListener('DOMContentLoaded', function() {
  // Get JSON data
  var chartLabels = JSON.parse(document.getElementById('chart-labels-data').textContent);
  var chartDatasets = JSON.parse(document.getElementById('chart-datasets-data').textContent);
  var statsData = JSON.parse(document.getElementById('stats-data').textContent);
  var eventsData = JSON.parse(document.getElementById('events-data').textContent);
  var indicatorsData = JSON.parse(document.getElementById('indicators-data').textContent);
  var defaultChartType = JSON.parse(document.getElementById('chart-type-data').textContent) || 'line';
  
  console.log("Default chart type:", defaultChartType);
  
  // Create charts for each metric
  var charts = {};
  
  // Metrics configuration
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
      displayName: 'Cardiovascular Load Index (CVL)',
      unit: ''
    },
    'sdnn': {
      color: '#8898aa',
      bgColor: 'rgba(136, 152, 170, 0.2)',
      displayName: 'Heart Rate Variability (SDNN)',
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
  
  // Function to format specific values according to the metric
  function formatMetricValue(value, metric) {
    if (metric === 'temperatura') {
      // For temperature, show variation with sign
      let sign = value >= 0 ? '+' : '';
      return `${sign}${value.toFixed(1)}°C`;
    } else if (['fc', 'pasos', 'spo2'].includes(metric)) {
      // Integer values for certain metrics
      return Math.round(value);
    } else {
      // One decimal for other metrics
      return value.toFixed(1);
    }
  }
  
  // FUNCTION: Update events table for each metric WITH PAGINATION
  function updateEventsTable(metric, events) {
    var tbody = document.getElementById(metric + '-events');
    var paginationContainer = document.getElementById(metric + '-pagination');
    if (!tbody) return;
    
    // Clear previous content
    tbody.innerHTML = '';
    if (paginationContainer) paginationContainer.innerHTML = '';
    
    // If no events, show message
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
    
    // Pagination configuration
    const eventsPerPage = 5;
    const totalPages = Math.ceil(events.length / eventsPerPage);
    let currentPage = 1;
    
    // Function to display events of the current page
    function displayEvents(page) {
      tbody.innerHTML = '';
      currentPage = page;
      
      const start = (page - 1) * eventsPerPage;
      const end = Math.min(start + eventsPerPage, events.length);
      const pageEvents = events.slice(start, end);
      
      pageEvents.forEach(event => {
        var tr = document.createElement('tr');
        
        // Time column
        var tdTime = document.createElement('td');
        tdTime.textContent = event.timestamp || 'N/A';
        tr.appendChild(tdTime);
        
        // Biometric value column with improved formatting
        var tdValue = document.createElement('td');
        var valueSpan = document.createElement('span');
        valueSpan.classList.add(event.is_abnormal ? 'text-warning' : 'text-success');
        
        // Format value according to metric type
        let formattedValue;
        if (metric === 'temperatura') {
          let sign = event.value >= 0 ? '+' : '';
          formattedValue = `${sign}${event.value.toFixed(1)}°C`;
        } else {
          formattedValue = `${event.value} ${window.metricConfig && window.metricConfig[metric]?.unit || ''}`;
        }
        
        // Add tooltip with abnormality reason
        if (event.is_abnormal) {
          let tooltipText = event.abnormal_reason || 'Abnormal value';
          valueSpan.innerHTML = `${formattedValue} 
                        <i class="fas fa-exclamation-triangle ml-1" data-toggle="tooltip" title="${tooltipText}"></i>`;
        } else {
          valueSpan.innerHTML = `${formattedValue} 
                        <i class="fas fa-check-circle ml-1" data-toggle="tooltip" title="Normal value"></i>`;
        }
        
        tdValue.appendChild(valueSpan);
        tr.appendChild(tdValue);
        
        // Activity column
        var tdActivity = document.createElement('td');
        tdActivity.innerHTML = `<strong>${event.activity_type || 'Unknown'}</strong>`;
        tr.appendChild(tdActivity);
        
        // Details column
        var tdDetails = document.createElement('td');
        if (event.details) {
          var detailsList = document.createElement('ul');
          detailsList.className = 'mb-0 pl-3';
          
          // Show only the 3 most relevant details
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
      
      // Reinitialize tooltips for new elements
      $('[data-toggle="tooltip"]').tooltip();
      
      // Update pagination controls
      updatePaginationControls();
    }
    
    // Function to update pagination controls
    function updatePaginationControls() {
      if (!paginationContainer) return;
      
      // Don't show pagination if there's only one page
      if (totalPages <= 1) {
        paginationContainer.innerHTML = '';
        
        // If there are events but not enough to paginate, show a simple counter
        if (events.length > 0) {
          const countDiv = document.createElement('div');
          countDiv.className = 'text-center text-muted mt-2';
          countDiv.innerHTML = `<small>${events.length} event${events.length !== 1 ? 's' : ''} in total</small>`;
          paginationContainer.appendChild(countDiv);
        }
        return;
      }
      
      paginationContainer.innerHTML = '';
      
      // Main container to center all pagination
      const paginationWrapper = document.createElement('div');
      paginationWrapper.className = 'd-flex flex-column align-items-center';
      
      // Pagination container with style
      const paginationNav = document.createElement('nav');
      paginationNav.setAttribute('aria-label', 'Events pagination');
      paginationNav.className = 'mb-2'; // Add bottom margin
      
      const paginationUl = document.createElement('ul');
      paginationUl.className = 'pagination pagination-sm justify-content-center';
      
      // Previous button
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
      
      // Page buttons (with simplification for many pages)
      const maxPageButtons = 5;
      let startPage = Math.max(1, currentPage - Math.floor(maxPageButtons / 2));
      let endPage = Math.min(totalPages, startPage + maxPageButtons - 1);
      
      if (endPage - startPage + 1 < maxPageButtons) {
        startPage = Math.max(1, endPage - maxPageButtons + 1);
      }
      
      // Add first page and ellipsis if necessary
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
      
      // Numbered pages
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
      
      // Add last page and ellipsis if necessary
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
      
      // Next button
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
      
      // Event counter below pagination
      const countDiv = document.createElement('div');
      countDiv.className = 'text-center text-muted';
      countDiv.innerHTML = `<small>${events.length} event${events.length !== 1 ? 's' : ''} in total</small>`;
      paginationWrapper.appendChild(countDiv);
      
      paginationContainer.appendChild(paginationWrapper);
    }
    
    // Show the first page of events
    displayEvents(1);
  }
  
  // Function to format time labels more readably
  function formatTimeLabel(timeStr) {
    if (!timeStr) return '';
    
    // Try different time formats
    try {
      // If it's a complete ISO timestamp
      if (timeStr.includes('T')) {
        const date = new Date(timeStr);
        if (!isNaN(date.getTime())) {
          return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
      }
      
      // If it's a standard timestamp format with space
      if (timeStr.includes(' ') && timeStr.length > 8) {
        const timePart = timeStr.split(' ')[1];
        if (timePart && timePart.includes(':')) {
          return timePart.substring(0, 5); // HH:MM
        }
      }
      
      // If it's already an hour, only return the first 5 characters (HH:MM)
      if (timeStr.includes(':') && timeStr.length >= 5) {
        return timeStr.substring(0, 5);
      }
    } catch (e) {
      console.warn("Error formatting time label:", e);
    }
    
    return timeStr; // Return the original if it can't be formatted
  }
  
  // NEW FUNCTION: Extract hourly timestamps for X axis
  function extractHourlyTimestamps(timestamps, totalDuration) {
    if (!timestamps || timestamps.length === 0) return [];

    try {
      // Try to convert to Date objects
      const dateTimes = timestamps
        .map(ts => {
          try {
            // Handle different formats
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

      // Sort the times
      dateTimes.sort((a, b) => a - b);

      // Get the initial and final time
      const startTime = dateTimes[0];
      const endTime = dateTimes[dateTimes.length - 1];
      
      // Total duration in hours
      const durationHours = (endTime - startTime) / (1000 * 60 * 60);
      
      // If the duration is less than 2 hours, use shorter intervals
      const hourInterval = durationHours < 2 ? 0.5 : 1;
      
      // Generate timestamps per hour
      const hourlyTimestamps = [];
      let currentTime = new Date(startTime);
      
      // Add the initial time
      hourlyTimestamps.push({
        timestamp: currentTime.toISOString(),
        label: formatTimeLabel(currentTime.toTimeString()),
        position: 0 // Relative position (0 = start)
      });
      
      // Add intermediate hours
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
      console.error("Error extracting hourly timestamps:", e);
      return [];
    }
  }

  // NEW FUNCTION: Optimize data visualization to show a complete view
  function createOptimizedView(originalData, originalLabels, eventIndices) {
    // If there is little data, show all of it
    if (originalData.length <= 1000) {
      return {
        data: originalData,
        indices: Array.from({ length: originalData.length }, (_, i) => i),
        labels: originalLabels
      };
    }

    // Create arrays for data and their corresponding indices
    const viewData = [];
    const viewIndices = [];
    const viewLabels = [];
    
    // Get hourly timestamps to improve visualization
    const hourlyTimestamps = extractHourlyTimestamps(originalLabels, originalData.length);
    
    // Convert event indices to a set for quick lookup
    const eventSet = new Set(eventIndices);
    
    // Always include the first and last point
    viewData.push(originalData[0]);
    viewIndices.push(0);
    viewLabels.push(originalLabels[0]);
    
    // Add all important event points
    for (const idx of eventIndices) {
      if (idx > 0 && idx < originalData.length - 1) {  // Avoid duplicating first/last point
        viewData.push(originalData[idx]);
        viewIndices.push(idx);
        viewLabels.push(originalLabels[idx]);
      }
    }
    
    // Add points at regular intervals
    const step = Math.max(1, Math.floor(originalData.length / 100));
    for (let i = step; i < originalData.length - 1; i += step) {
      // Avoid duplicating event points
      if (!eventSet.has(i)) {
        viewData.push(originalData[i]);
        viewIndices.push(i);
        viewLabels.push(originalLabels[i]);
      }
    }
    
    // Always include the last point if it's not already included
    if (!eventSet.has(originalData.length - 1)) {
      viewData.push(originalData[originalData.length - 1]);
      viewIndices.push(originalData.length - 1);
      viewLabels.push(originalLabels[originalData.length - 1]);
    }
    
    // Sort by index to maintain chronological order
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
  
  // Add zoom instructions banner
  function addZoomInstructionsBanner() {
    // Check if it already exists
    if (document.getElementById('zoom-instructions-banner')) return;
    
    // Find the first container to insert it
    const container = document.querySelector('.container-fluid');
    if (!container) return;
    
    const banner = document.createElement('div');
    banner.id = 'zoom-instructions-banner';
    banner.className = 'alert alert-info alert-dismissible fade show mb-4';
    banner.role = 'alert';
    banner.innerHTML = `
      <div class="d-flex align-items-center">
        <i class="fas fa-mouse mr-2"></i>
        <div><strong>Tip:</strong> You can zoom in on the charts by scrolling the mouse wheel over them</div>
      </div>
      <button type="button" class="close" data-dismiss="alert" aria-label="Close">
        <span aria-hidden="true">&times;</span>
      </button>
    `;
    
    // Insert at the beginning of the container
    container.insertBefore(banner, container.firstChild);
  }
  
  // Add the instructions banner only once
  addZoomInstructionsBanner();
  
  // Function to add zoom message and reset button below the chart
  function addZoomMessage(chartContainer, chartInstance) {
    // Check if it already exists
    if (chartContainer.querySelector('.zoom-message-container')) return;
    
    // Create the container for the zoom message
    const zoomMessageContainer = document.createElement('div');
    zoomMessageContainer.className = 'zoom-message-container text-center mt-2';
    
    // Create the message
    const zoomMessage = document.createElement('div');
    zoomMessage.className = 'zoom-message text-muted small';
    zoomMessage.innerHTML = '<i class="fas fa-mouse mr-1"></i> Use the mouse wheel to zoom';
    
    // Create the reset button
    const resetButton = document.createElement('button');
    resetButton.type = 'button';
    resetButton.className = 'btn btn-sm btn-outline-primary ml-2';
    resetButton.innerHTML = '<i class="fas fa-undo mr-1"></i> Reset zoom';
    resetButton.onclick = function() {
      // Try different methods to reset the zoom
      try {
        if (chartInstance.resetZoom) {
          chartInstance.resetZoom();
        } else if (chartInstance.scales && chartInstance.scales.x) {
          // Alternative for newer Chart.js
          chartInstance.scales.x.options.min = undefined;
          chartInstance.scales.x.options.max = undefined;
          chartInstance.scales.y.options.min = undefined;
          chartInstance.scales.y.options.max = undefined;
          chartInstance.update();
        } else {
          // Last option: recreate the chart
          chartInstance.update();
        }
      } catch (error) {
        console.error("Error resetting zoom:", error);
      }
    };
    
    // Add message and button to the container
    const wrapper = document.createElement('div');
    wrapper.className = 'd-flex align-items-center justify-content-center';
    wrapper.appendChild(zoomMessage);
    wrapper.appendChild(resetButton);
    
    zoomMessageContainer.appendChild(wrapper);
    
    // Insert the container after the canvas
    chartContainer.appendChild(zoomMessageContainer);
  }
  
  // Create charts for each dataset with visual improvements
  chartDatasets.forEach(dataset => {
    var metric = dataset.metric;
    if (!metric) return;
    
    var ctx = document.getElementById(metric + '-chart');
    if (!ctx) return;

    // Process events for this metric
    const events = eventsData[metric] || [];
    
    // Extract important event indices
    const eventIndices = events.map(event => event.index || 0);
    
    // Update the events table
    updateEventsTable(metric, events);
    
    // IMPROVED SOLUTION: Preprocess labels for X axis
    const processedLabels = Array.isArray(chartLabels) ? chartLabels.map(label => {
      if (typeof label === 'string') {
        return formatTimeLabel(label);
      }
      return label;
    }) : [];
    
    // Create optimized view for visualization
    const optimizedView = createOptimizedView(
      dataset.data, 
      processedLabels.length === dataset.data.length ? processedLabels : Array(dataset.data.length).fill(''),
      eventIndices
    );
    
    const optimizedData = optimizedView.data;
    const optimizedIndices = optimizedView.indices;
    const optimizedLabels = optimizedView.labels;
    
    // Get the configured chart type
    var chartType = dataset.chart_type || defaultChartType || 'line';
    
    // Configuration for points on the chart
    var pointRadius = Array(optimizedData.length).fill(2);
    var pointBackgroundColor = Array(optimizedData.length).fill(dataset.borderColor);
    var pointBorderColor = Array(optimizedData.length).fill('#fff');
    var pointBorderWidth = Array(optimizedData.length).fill(1);
    
    // Highlight important event points
    events.forEach(event => {
      const idx = optimizedIndices.indexOf(event.index);
      if (idx !== -1) {
        pointRadius[idx] = event.is_abnormal ? 6 : 5;
        pointBackgroundColor[idx] = event.is_abnormal ? '#fb6340' : '#2dce89';
        pointBorderColor[idx] = '#ffffff';
        pointBorderWidth[idx] = 2;
      }
    });
    
    // Information for tooltips (context of each point)
    var activityInfo = Array(optimizedData.length).fill(null);
    
    // Associate event information with their corresponding points
    events.forEach(event => {
      const idx = optimizedIndices.indexOf(event.index);
      if (idx !== -1) {
        activityInfo[idx] = {
          timestamp: event.timestamp || '',
          activity_type: event.activity_type || 'Activity',
          is_abnormal: event.is_abnormal || false,
          abnormal_reason: event.abnormal_reason || 'Normal',
          details: event.details || {},
          isEvent: true
        };
      }
    });
    
    // Complete information for points that don't have associated events
    for (let i = 0; i < optimizedData.length; i++) {
      if (!activityInfo[i]) {
        activityInfo[i] = {
          timestamp: optimizedLabels[i] || '',
          activity_type: '',  // Removed "Data point"
          is_abnormal: false,
          abnormal_reason: 'Normal value',
          details: {},
          isEvent: false
        };
      }
    }
    
    // VISUAL IMPROVEMENT: Extract hours for annotations and reference lines
    const hourlyTimestamps = extractHourlyTimestamps(chartLabels, dataset.data.length);
    
    // VISUAL IMPROVEMENT: Prepare annotations and reference lines
    const annotations = {};
    
    // Add vertical lines for each hour
    hourlyTimestamps.forEach((hourPoint, idx) => {
      // Find the nearest index in optimized data
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
    
    // IMPROVEMENT: Add horizontal lines for reference values according to metric
    if (metric === 'fc') {
      // Line for maximum normal heart rate at rest
      annotations['fc-normal-max'] = {
        type: 'line',
        scaleID: 'y',
        value: 100,
        borderColor: 'rgba(245, 54, 92, 0.5)',
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          content: 'Max normal HR',
          enabled: true,
          position: 'end',
          backgroundColor: 'rgba(245, 54, 92, 0.7)',
          color: '#ffffff',
          font: { size: 10 }
        }
      };
    } else if (metric === 'spo2') {
      // Line for optimal oxygen saturation level
      annotations['spo2-normal'] = {
        type: 'line',
        scaleID: 'y',
        value: 95,
        borderColor: 'rgba(29, 140, 248, 0.5)',
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          content: 'Optimal SpO₂',
          enabled: true,
          position: 'end',
          backgroundColor: 'rgba(29, 140, 248, 0.7)',
          color: '#ffffff',
          font: { size: 10 }
        }
      };
    }
    
    // IMPROVEMENT: Chart configuration with better visualization
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
                
                // Show only the time as title
                if (info && info.timestamp) {
                  return info.timestamp;
                }
                return optimizedLabels[dataIndex] || 'Time not available';
              },
              label: function(context) {
                const dataIndex = context.dataIndex;
                let lines = [];
                
                // Line 1: Basic value with units
                var basicLabel = context.dataset.label || '';
                if (basicLabel) basicLabel += ': ';
                
                if (context.parsed.y !== null) {
                  // Special formatting for temperature
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
                
                // Show relevant information
                const info = activityInfo[dataIndex];
                
                if (info) {
                  // Only show activity if it's a real event with defined activity
                  if (info.isEvent && info.activity_type) {
                    lines.push(`Activity: ${info.activity_type}`);
                  }
                  
                  // Status (always show, with reason if abnormal)
                  if (info.is_abnormal) {
                    lines.push(`Status: Abnormal (${info.abnormal_reason || 'No details'})`);
                  } else {
                    lines.push('Status: Normal');
                  }
                  
                  // Application and action details (maximum 2 details)
                  if (info.details) {
                    // Prioritize showing the application
                    if (info.details['app']) {
                      lines.push(`App: ${info.details['app']}`);
                    }
                    
                    // Look for relevant action information
                    const actionKeys = ['action', 'Element Text', 'Element Type', 'Input'];
                    for (const key of actionKeys) {
                      if (info.details[key]) {
                        lines.push(`${key === 'action' ? 'Action' : key}: ${info.details[key]}`);
                        break; // Only show one action
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
                modifierKey: null  // No modifier key required (Ctrl)
              },
              pinch: { 
                enabled: true 
              },
              mode: 'x',
              onZoom: function() {
                // This callback is important for the zoom event to be properly registered
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
              maxRotation: 45, // Rotate labels to avoid overlap
              minRotation: 45, // Keep a constant rotation
              autoSkip: true, // Enable automatic label skipping
              autoSkipPadding: 15, // Minimum space between labels
              callback: function(val, index) {
                // Show labels at intervals to avoid overloading
                if (optimizedLabels.length > 50) {
                  // For many data points, show fewer labels
                  if (index % 5 === 0) {
                    return this.getLabelForValue(val);
                  }
                  return '';
                }
                return this.getLabelForValue(val);
              },
              color: 'rgba(0, 0, 0, 0.75)', // Make labels more visible
              font: {
                size: 10,
                weight: 'bold' // More prominent font
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
    
    // Specific adjustments for area charts
    if (chartType === 'area') {
      // Increase background color opacity for area
      if (config.data.datasets[0].backgroundColor.includes('rgba')) {
        config.data.datasets[0].backgroundColor = config.data.datasets[0].backgroundColor.replace(
          /rgba\((\d+),\s*(\d+),\s*(\d+),\s*[\d.]+\)/,
          'rgba($1, $2, $3, 0.5)'
        );
      } else {
        // If not rgba, add opacity
        var color = config.data.datasets[0].backgroundColor;
        if (color.startsWith('#')) {
          // Convert HEX to RGBA
          var r = parseInt(color.slice(1, 3), 16);
          var g = parseInt(color.slice(3, 5), 16);
          var b = parseInt(color.slice(5, 7), 16);
          config.data.datasets[0].backgroundColor = `rgba(${r}, ${g}, ${b}, 0.5)`;
        }
      }
    }
    
    // Specific optimizations by metric and chart type
    if (metric === 'fc' && chartType === 'bar') {
      config.data.datasets[0].backgroundColor = 'rgba(245, 54, 92, 0.6)';
    }
    
    if (metric === 'pasos' && chartType === 'line') {
      config.data.datasets[0].stepped = true;
    }
    
    // Specific adjustments for temperature
    if (metric === 'temperatura') {
      try {
        // Special colors for positive and negative temperatures
        const gradientAboveZero = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
        gradientAboveZero.addColorStop(0, 'rgba(251, 99, 64, 0.8)');
        gradientAboveZero.addColorStop(1, 'rgba(251, 99, 64, 0.1)');
        
        const gradientBelowZero = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
        gradientBelowZero.addColorStop(0, 'rgba(94, 114, 228, 0.1)');
        gradientBelowZero.addColorStop(1, 'rgba(94, 114, 228, 0.8)');
        
        // Apply colors according to chart type
        if (chartType === 'area') {
          // For area, use gradients according to value
          config.data.datasets[0].backgroundColor = function(context) {
            const value = context.raw;
            return value >= 0 ? gradientAboveZero : gradientBelowZero;
          };
        } else if (chartType === 'bar') {
          // For bars, color according to value
          config.data.datasets[0].backgroundColor = function(context) {
            const value = context.raw;
            return value >= 0 ? 'rgba(251, 99, 64, 0.6)' : 'rgba(94, 114, 228, 0.6)';
          };
        }
      } catch (error) {
        console.warn("Error configuring gradients for temperature:", error);
      }
    }
    
    // IMPROVEMENT: Load necessary plugins dynamically and then create the chart
    function loadPlugins() {
      return new Promise((resolve, reject) => {
        // List of plugins we need
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
        
        // Register plugins that are already loaded
        const loadedPlugins = {};
        plugins.forEach(plugin => {
          loadedPlugins[plugin.name] = window[plugin.name] !== undefined;
        });
        
        // Function to load a plugin
        function loadPlugin(plugin) {
          return new Promise((resolvePlugin, rejectPlugin) => {
            // If already loaded, resolve immediately
            if (loadedPlugins[plugin.name]) {
              resolvePlugin();
              return;
            }
            
            // Check dependencies
            if (plugin.requires) {
              for (const dep of plugin.requires) {
                if (!loadedPlugins[dep]) {
                  rejectPlugin(`Plugin ${plugin.name} requires ${dep} which is not loaded`);
                  return;
                }
              }
            }
            
            // Load the script
            const script = document.createElement('script');
            script.src = plugin.url;
            script.onload = () => {
              loadedPlugins[plugin.name] = true;
              resolvePlugin();
            };
            script.onerror = () => rejectPlugin(`Error loading ${plugin.name}`);
            document.head.appendChild(script);
          });
        }
        
        // Load plugins in specific order (hammer first, then the rest)
        loadPlugin(plugins.find(p => p.name === 'hammer'))
          .then(() => {
            // Load the remaining plugins in parallel
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
    
    // Create the chart with the necessary plugins
    loadPlugins().then(() => {
      // Register plugins globally for Chart.js if needed
      if (window['chartjs-plugin-zoom'] && typeof Chart.register === 'function') {
        Chart.register(window['chartjs-plugin-zoom']);
      }
      
      if (window['chartjs-plugin-annotation'] && typeof Chart.register === 'function') {
        Chart.register(window['chartjs-plugin-annotation']);
      }
      
      // Create the chart once all plugins are loaded
      charts[metric] = new Chart(ctx, config);
      
      // Find the chart container to add the zoom message
      const chartContainer = ctx.parentNode;
      addZoomMessage(chartContainer, charts[metric]);
      
    }).catch(err => {
      console.error('Error loading plugins:', err);
      
      // If plugin loading fails, create a simplified version of the chart
      delete config.options.plugins.annotation;
      delete config.options.plugins.zoom;
      
      charts[metric] = new Chart(ctx, config);
    });
  });
  
  // Tab navigation using pure JavaScript
  var tabLinks = document.querySelectorAll('.custom-tabs .nav-link');
  var tabContents = document.querySelectorAll('.tab-pane');
  
  // Function to show a specific tab
  function showTab(tabId) {
    // Hide all tab contents
    tabContents.forEach(function(content) {
      content.classList.remove('show', 'active');
    });
    
    // Deactivate all tabs
    tabLinks.forEach(function(link) {
      link.classList.remove('active');
      link.setAttribute('aria-selected', 'false');
    });
    
    // Activate the selected tab
    var selectedTab = document.getElementById(tabId + '-tab');
    if (selectedTab) {
      selectedTab.classList.add('active');
      selectedTab.setAttribute('aria-selected', 'true');
    }
    
    // Show the tab content
    var selectedContent = document.getElementById(tabId + '-content');
    if (selectedContent) {
      selectedContent.classList.add('show', 'active');
    }
    
    // Resize the chart if it exists
    if (charts[tabId]) {
      setTimeout(function() {
        charts[tabId].resize();
      }, 50);
    }
  }
  
  // Assign click events to tabs
  tabLinks.forEach(function(link) {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      var tabId = this.id.replace('-tab', '');
      showTab(tabId);
    });
  });
  
  // Activate the first tab by default
  if (tabLinks.length > 0) {
    var firstTabId = tabLinks[0].id.replace('-tab', '');
    showTab(firstTabId);
  }
  
  // Apply colors to indicators
  for (const [key, indicator] of Object.entries(indicatorsData)) {
    const card = document.getElementById('indicator-' + key);
    if (card && indicator.color) {
      // Use predefined classes
      if (indicator.color === 'primary') card.classList.add('border-primary');
      else if (indicator.color === 'danger') card.classList.add('border-danger');
      else if (indicator.color === 'warning') card.classList.add('border-warning');
      else if (indicator.color === 'success') card.classList.add('border-success');
      else if (indicator.color === 'info') card.classList.add('border-info');
      else if (indicator.color === 'purple') card.classList.add('border-purple');
      else card.style.borderColor = indicator.color; // Fallback to inline style
    }
  }
  
  // Remove any unnecessary scrollbar
  document.querySelectorAll('.card-body, .nav-tabs-container, .custom-tabs, .tab-content').forEach(function(element) {
    element.classList.add('no-scroll');
  });
  
  // Fix dropdowns
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
  
  // Observe changes in DOM
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
  
  // Initialize tooltips
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