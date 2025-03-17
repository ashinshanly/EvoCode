/**
 * EvoCode Web Interface
 * Main JavaScript functionality for the web interface
 */

// Global variables
let socket;
let editor;
let evolutionChart;
let processId;
let statusInterval;
let elapsedTimeInterval;
let startTime;

// Sample code for quick testing
const SAMPLE_CODE = `def bubble_sort(arr):
    """
    A simple bubble sort implementation to be optimized.
    
    Args:
        arr: List of integers to sort
        
    Returns:
        Sorted list of integers
    """
    # Make a copy to avoid modifying the original
    result = arr.copy()
    n = len(result)
    
    # Bubble sort algorithm
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    
    return result
`;

// DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize CodeMirror editor
    initializeEditor();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize highlight.js
    hljs.highlightAll();
    
    // Initialize the evolution chart
    initializeEvolutionChart();
    
    // Add the pulse animation to the document
    const style = document.createElement('style');
    style.innerHTML = `
        @keyframes pulse {
            0% { box-shadow: 0 0 15px rgba(25, 135, 84, 0.5); }
            50% { box-shadow: 0 0 25px rgba(25, 135, 84, 0.8); }
            100% { box-shadow: 0 0 15px rgba(25, 135, 84, 0.5); }
        }
    `;
    document.head.appendChild(style);
});

/**
 * Initialize the CodeMirror editor
 */
function initializeEditor() {
    const codeEditorElem = document.getElementById('codeEditor');
    
    if (codeEditorElem) {
        editor = CodeMirror.fromTextArea(codeEditorElem, {
            mode: 'python',
            theme: 'monokai',
            lineNumbers: true,
            indentUnit: 4,
            tabSize: 4,
            indentWithTabs: false,
            lineWrapping: true,
            autoCloseBrackets: true,
            matchBrackets: true,
            extraKeys: {
                'Tab': (cm) => cm.execCommand('indentMore'),
                'Shift-Tab': (cm) => cm.execCommand('indentLess'),
            }
        });
    }
}

/**
 * Setup all event listeners for the UI
 */
function setupEventListeners() {
    // Form submission
    const codeForm = document.getElementById('codeForm');
    if (codeForm) {
        codeForm.addEventListener('submit', handleFormSubmit);
    }
    
    // Load sample code button
    const sampleCodeBtn = document.getElementById('sampleCodeBtn');
    if (sampleCodeBtn) {
        sampleCodeBtn.addEventListener('click', () => {
            editor.setValue(SAMPLE_CODE);
            document.getElementById('functionName').value = 'bubble_sort';
        });
    }
    
    // Cancel button
    const cancelBtn = document.getElementById('cancelBtn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', cancelEvolution);
    }
    
    // Restart button
    const restartBtn = document.getElementById('restartBtn');
    if (restartBtn) {
        restartBtn.addEventListener('click', restartEvolution);
    }
    
    // Download button
    const downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadEvolvedCode);
    }
    
    // File upload handling
    const fileUpload = document.getElementById('fileUpload');
    if (fileUpload) {
        fileUpload.addEventListener('change', handleFileUpload);
    }
}

/**
 * Handle file upload
 */
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Read the file contents
    const reader = new FileReader();
    reader.onload = function(e) {
        const content = e.target.result;
        editor.setValue(content);
        
        // Try to extract the function name from the file
        const functionNameMatch = content.match(/def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/);
        if (functionNameMatch && functionNameMatch[1]) {
            document.getElementById('functionName').value = functionNameMatch[1];
        }
    };
    reader.readAsText(file);
}

/**
 * Handle form submission
 */
function handleFormSubmit(event) {
    event.preventDefault();
    
    // Update the code textarea with the editor content
    editor.save();
    
    // Create form data
    const formData = new FormData(event.target);
    
    // Show the results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('statusMessage').innerHTML = '<i class="bi bi-info-circle me-2"></i>Submitting code...';
    
    // Reset chart
    if (evolutionChart) {
        evolutionChart.destroy();
    }
    
    // Initialize chart
    initializeEvolutionChart();
    
    // Submit the form
    fetch('/api/submit', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Store the process ID
        processId = data.process_id;
        
        // Initialize socket connection
        initializeSocket(processId);
        
        // Start status polling
        startStatusPolling(processId);
        
        // Update UI
        document.getElementById('statusMessage').innerHTML = `
            <i class="bi bi-info-circle me-2"></i>
            Evolution started for function <strong>${data.function_name}</strong>
        `;
        
        // Record start time
        startTime = new Date();
        
        // Start elapsed time counter
        startElapsedTimeCounter();
    })
    .catch(error => {
        showError('Error submitting code: ' + error.message);
    });
}

/**
 * Initialize Socket.IO connection
 */
function initializeSocket(processId) {
    // Close existing socket if any
    if (socket) {
        socket.disconnect();
    }
    
    // Update status to show connection attempt
    updateStatus("Connecting to server...", "info");
    
    // Connect to Socket.IO server
    console.log("Attempting to connect to Socket.IO server...");
    socket = io({
        reconnectionAttempts: 5,
        timeout: 10000,
        reconnectionDelay: 1000
    });
    
    // Debug all socket events
    socket.onAny((event, ...args) => {
        console.log(`Socket event received: ${event}`, args);
    });
    
    // Connection established
    socket.on('connect', () => {
        console.log('Socket connected with ID:', socket.id);
        updateStatus("Connected to server. Subscribing to updates...", "info");
        
        // Subscribe to updates for this process
        socket.emit('subscribe', { process_id: processId });
    });
    
    // Connection error
    socket.on('connect_error', (error) => {
        console.error('Socket.IO connection error:', error);
        updateStatus(`Connection error: ${error.message}. Retrying...`, 'warning');
    });
    
    // Handle subscription confirmation
    socket.on('subscribed', (data) => {
        console.log('Subscribed to updates for process:', data.process_id);
        updateStatus(`Subscribed to evolution process ${data.process_id.substring(0,8)}...`, "info");
    });
    
    // Handle evolution start
    socket.on('evolution_started', (data) => {
        console.log('Evolution started:', data);
        updateStatus(`Evolution process started for ${data.function_name}`, "info");
    });
    
    // Handle global events (fallback)
    socket.on('global_evolution_started', (data) => {
        console.log('Global evolution started event:', data);
        if (data.process_id === processId) {
            updateStatus(`Evolution process started (global event)`, "info");
        }
    });
    
    // Handle function loading
    socket.on('loading_function', (data) => {
        console.log('Loading function:', data);
        updateStatus(data.message, 'info');
    });
    
    // Handle test case generation
    socket.on('generating_test_cases', (data) => {
        console.log('Generating test cases:', data);
        updateStatus(data.message, 'info');
    });
    
    // Handle test cases generated
    socket.on('test_cases_generated', (data) => {
        console.log('Test cases generated:', data);
        updateStatus(`Generated ${data.count} test cases`, 'info');
    });
    
    // Handle evolution progress
    socket.on('evolution_in_progress', (data) => {
        console.log('Evolution in progress:', data);
        updateStatus(data.message, 'info');
        document.getElementById('generationCounter').textContent = `0/${data.generations}`;
    });
    
    // Handle generation complete
    socket.on('generation_complete', (data) => {
        console.log('Generation complete:', data);
        
        // Update progress bar
        const progressBar = document.getElementById('progressBar');
        const progress = Math.round(data.progress);
        progressBar.style.width = `${progress}%`;
        progressBar.textContent = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
        
        // Update generation counter
        document.getElementById('generationCounter').textContent = `${data.generation}/${data.generations}`;
        
        // Update best fitness
        document.getElementById('bestFitness').textContent = data.best_fitness.toFixed(6);
        
        // Update chart
        updateEvolutionChart(data);
        
        // Update status message
        updateStatus(`Evolution in progress: Generation ${data.generation}/${data.generations} complete`, 'info');
    });
    
    // Set up a keepalive ping to check if the server is responsive
    setInterval(() => {
        if (socket.connected) {
            const startTime = Date.now();
            socket.emit('ping', {}, () => {
                const latency = Date.now() - startTime;
                console.log(`Socket ping: ${latency}ms`);
            });
        }
    }, 5000);
    
    // Set up a watchdog to detect if we're not getting updates
    let lastUpdateTime = Date.now();
    const watchdogInterval = setInterval(() => {
        const currentTime = Date.now();
        const elapsedSinceLastUpdate = (currentTime - lastUpdateTime) / 1000;
        
        // If no updates for 30 seconds, show a warning
        if (elapsedSinceLastUpdate > 30) {
            console.warn(`No updates received for ${elapsedSinceLastUpdate.toFixed(0)} seconds`);
            updateStatus(`No updates received for ${elapsedSinceLastUpdate.toFixed(0)} seconds. The evolution process may be stuck.`, 'warning');
            
            // After 60 seconds with no updates, try to fetch status directly
            if (elapsedSinceLastUpdate > 60 && processId) {
                console.warn("Trying to fetch status directly via API");
                fetch(`/api/status/${processId}`)
                    .then(response => response.json())
                    .then(data => {
                        console.log("Status API response:", data);
                        updateStatus(`Status check: ${data.is_running ? 'Still running' : 'Not running'}. Generation: ${data.current_generation}/${data.total_generations}`, 
                                   data.error ? 'danger' : 'warning');
                        
                        // If the process is complete or has an error, clear the watchdog
                        if (data.is_complete || data.error || data.is_cancelled) {
                            clearInterval(watchdogInterval);
                            
                            // If complete, fetch the results
                            if (data.is_complete) {
                                fetchFinalResults(processId);
                            }
                        }
                    })
                    .catch(error => {
                        console.error("Error fetching status:", error);
                        updateStatus(`Error checking status: ${error.message}`, 'danger');
                    });
            }
        }
    }, 10000); // Check every 10 seconds
    
    // Update the lastUpdateTime whenever we receive any evolution-related event
    const updateEvents = [
        'evolution_started', 'loading_function', 'generating_test_cases', 
        'test_cases_generated', 'evolution_in_progress', 'generation_complete',
        'mutation_update', 'benchmarking', 'evolution_complete',
        'visualization_data'
    ];
    
    updateEvents.forEach(eventName => {
        socket.on(eventName, () => {
            lastUpdateTime = Date.now();
        });
    });
    
    // Handle mutation updates
    socket.on('mutation_update', (data) => {
        console.log('Mutation update:', data);
        updateMutationsDisplay(data.mutations);
    });
    
    // Handle benchmarking
    socket.on('benchmarking', (data) => {
        console.log('Benchmarking:', data);
        updateStatus(data.message, 'info');
    });
    
    // Handle evolution complete
    socket.on('evolution_complete', (data) => {
        console.log('Evolution complete:', data);
        
        // Clear watchdog interval
        if (typeof watchdogInterval !== 'undefined') {
            clearInterval(watchdogInterval);
        }
        
        // Stop elapsed time counter
        stopElapsedTimeCounter();
        
        // Update status
        updateStatus(`${data.message}. Performance improvement: ${data.performance_improvement.toFixed(2)}%`, 'success');
        
        // Stop status polling
        if (statusInterval) {
            clearInterval(statusInterval);
        }
        
        // Get final results
        fetchFinalResults(processId);
        
        // Show final results section
        document.getElementById('finalResultsSection').style.display = 'block';
    });
    
    // Handle global evolution complete (fallback)
    socket.on('global_evolution_complete', (data) => {
        console.log('Global evolution complete event:', data);
        if (data.process_id === processId) {
            // Clear watchdog interval
            if (typeof watchdogInterval !== 'undefined') {
                clearInterval(watchdogInterval);
            }
            
            // Stop elapsed time counter
            stopElapsedTimeCounter();
            
            updateStatus(`Evolution complete (global event)`, 'success');
            
            if (statusInterval) {
                clearInterval(statusInterval);
            }
            
            fetchFinalResults(processId);
            
            // Show final results section
            document.getElementById('finalResultsSection').style.display = 'block';
        }
    });
    
    // Handle visualization data
    socket.on('visualization_data', (data) => {
        console.log('Visualization data received:', data);
        if (data.chart_data) {
            updateEvolutionChartWithFullData(data.chart_data);
        }
    });
    
    // Also listen for global visualization events as fallback
    socket.on('global_visualization_data', (data) => {
        console.log('Global visualization data received:', data);
        if (data.process_id === processId && data.chart_data) {
            console.log('Updating chart with global data');
            updateEvolutionChartWithFullData(data.chart_data);
        }
    });
    
    // Handle performance data
    socket.on('performance_data', (data) => {
        console.log('Performance data received:', data);
        if (data.performance) {
            updatePerformanceDisplay(data.performance);
        }
    });
    
    // Handle global performance data as fallback
    socket.on('global_performance_data', (data) => {
        console.log('Global performance data received:', data);
        if (data.process_id === processId && data.performance) {
            updatePerformanceDisplay(data.performance);
        }
    });
    
    // Handle evolution errors
    socket.on('evolution_error', (data) => {
        console.error('Evolution error:', data);
        
        // Clear watchdog interval
        clearInterval(watchdogInterval);
        
        showError(data.message);
        
        // Show traceback if available
        if (data.traceback) {
            console.error('Error traceback:', data.traceback);
        }
        
        // Stop status polling
        clearInterval(statusInterval);
    });
    
    // Handle global evolution error (fallback)
    socket.on('global_evolution_error', (data) => {
        console.error('Global evolution error event:', data);
        if (data.process_id === processId) {
            clearInterval(watchdogInterval);
            showError(`Evolution error: ${data.message}`);
            clearInterval(statusInterval);
        }
    });
    
    // Handle cancellation
    socket.on('evolution_cancelled', (data) => {
        console.log('Evolution cancelled:', data);
        
        // Clear watchdog interval
        clearInterval(watchdogInterval);
        
        updateStatus(data.message, 'warning');
        
        // Stop status polling
        clearInterval(statusInterval);
    });
    
    // Handle global cancellation (fallback)
    socket.on('global_evolution_cancelled', (data) => {
        console.log('Global evolution cancelled event:', data);
        if (data.process_id === processId) {
            clearInterval(watchdogInterval);
            updateStatus('Evolution process was cancelled', 'warning');
            clearInterval(statusInterval);
        }
    });
    
    // Handle disconnection
    socket.on('disconnect', (reason) => {
        console.log('Socket disconnected:', reason);
        
        if (reason === 'io server disconnect') {
            // The server has forcefully disconnected
            updateStatus('Disconnected from server. The server may be restarting.', 'warning');
            
            // Try to reconnect
            socket.connect();
        } else if (reason === 'transport close') {
            // The connection was closed
            updateStatus('Connection to server lost. Attempting to reconnect...', 'warning');
        } else {
            updateStatus(`Disconnected from server: ${reason}. Attempting to reconnect...`, 'warning');
        }
    });
    
    // Handle reconnection
    socket.on('reconnect', (attemptNumber) => {
        console.log('Socket reconnected after', attemptNumber, 'attempts');
        updateStatus('Reconnected to server', 'success');
        
        // Resubscribe
        socket.emit('subscribe', { process_id: processId });
    });
    
    // Handle reconnect failure
    socket.on('reconnect_failed', () => {
        console.error('Socket reconnection failed');
        updateStatus('Failed to reconnect to server. Please refresh the page.', 'danger');
    });
}

/**
 * Update the status message
 */
function updateStatus(message, type = 'info') {
    const statusMessage = document.getElementById('statusMessage');
    
    if (statusMessage) {
        let icon;
        
        switch (type) {
            case 'success':
                icon = 'bi-check-circle';
                statusMessage.className = 'alert alert-success';
                break;
            case 'warning':
                icon = 'bi-exclamation-triangle';
                statusMessage.className = 'alert alert-warning';
                break;
            case 'danger':
            case 'error':
                icon = 'bi-x-circle';
                statusMessage.className = 'alert alert-danger';
                break;
            case 'info':
            default:
                icon = 'bi-info-circle';
                statusMessage.className = 'alert alert-info';
                break;
        }
        
        statusMessage.innerHTML = `<i class="bi ${icon} me-2"></i>${message}`;
    }
}

/**
 * Show error message
 */
function showError(message) {
    updateStatus(message, 'danger');
}

/**
 * Start polling for status updates
 */
function startStatusPolling(processId) {
    // Clear any existing interval
    if (statusInterval) {
        clearInterval(statusInterval);
    }
    
    // Poll every 2 seconds
    statusInterval = setInterval(() => {
        fetch(`/api/status/${processId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    clearInterval(statusInterval);
                    showError(data.error);
                    return;
                }
                
                // If process completed, fetch final results
                if (data.is_complete) {
                    clearInterval(statusInterval);
                    fetchFinalResults(processId);
                }
                
                // If process cancelled, show message
                if (data.is_cancelled) {
                    clearInterval(statusInterval);
                    updateStatus('Evolution process was cancelled.', 'warning');
                }
            })
            .catch(error => {
                console.error('Error polling status:', error);
            });
    }, 2000);
}

/**
 * Clean up any running timers when the page is unloaded
 */
window.addEventListener('beforeunload', () => {
    if (statusInterval) {
        clearInterval(statusInterval);
    }
    if (elapsedTimeInterval) {
        clearInterval(elapsedTimeInterval);
    }
});

/**
 * Function to stop the elapsed time counter and set the final time
 */
function stopElapsedTimeCounter() {
    if (elapsedTimeInterval) {
        clearInterval(elapsedTimeInterval);
        elapsedTimeInterval = null;
        console.log('Elapsed time counter stopped');
        
        // If we have a final time, make sure it's displayed correctly
        if (document.getElementById('finalResultsSection').style.display !== 'none') {
            const totalTimeElem = document.getElementById('totalTime');
            if (totalTimeElem && totalTimeElem.textContent !== '0s') {
                const elapsedTimeElem = document.getElementById('timeElapsed');
                if (elapsedTimeElem) {
                    elapsedTimeElem.textContent = totalTimeElem.textContent;
                }
            }
        }
    }
}

/**
 * Function to make code displays refresh and update
 */
function refreshCodeDisplay() {
    // Force highlight.js to refresh code blocks
    document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
}

/**
 * Start elapsed time counter
 */
function startElapsedTimeCounter() {
    // Clear any existing interval
    stopElapsedTimeCounter();
    
    // Update every second
    elapsedTimeInterval = setInterval(() => {
        const elapsedTimeElem = document.getElementById('timeElapsed');
        
        if (elapsedTimeElem && startTime) {
            const elapsed = Math.floor((new Date() - startTime) / 1000);
            elapsedTimeElem.textContent = formatTime(elapsed);
        }
    }, 1000);
    
    console.log('Elapsed time counter started');
}

/**
 * Format time in seconds to MM:SS
 */
function formatTime(seconds) {
    if (seconds < 60) {
        return `${seconds}s`;
    }
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    return `${minutes}m ${remainingSeconds}s`;
}

/**
 * Initialize the evolution chart
 */
function initializeEvolutionChart() {
    const ctx = document.getElementById('evolutionChart').getContext('2d');
    
    // Clear any existing chart
    if (evolutionChart) {
        evolutionChart.destroy();
    }
    
    console.log('Initializing evolution chart');
    
    // Initialize with empty data
    evolutionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Max Fitness',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Avg Fitness',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${parseFloat(context.raw).toFixed(6)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    ticks: {
                        callback: function(value) {
                            return parseFloat(value).toFixed(4);
                        }
                    },
                    // Start with a very small scale to test if values close to zero are visible
                    min: 0,
                    max: 1,
                    suggestedMax: 1
                }
            },
            animation: {
                duration: 500
            }
        }
    });
    
    // Add some debug data to verify the chart works - using very small values to test rendering
    setTimeout(() => {
        const debugData = {
            generations: [1, 2, 3],
            max_fitness: [0.1, 0.2, 0.3],
            avg_fitness: [0.05, 0.1, 0.15]
        };
        console.log('Adding debug data to chart:', debugData);
        updateEvolutionChartWithFullData(debugData);
    }, 1000);
}

/**
 * Update the evolution chart with new data
 */
function updateEvolutionChart(data) {
    if (!evolutionChart) {
        console.warn('Chart not initialized, initializing now');
        initializeEvolutionChart();
    }
    
    try {
        console.log('Updating chart with single data point:', data);
        
        // Ensure data values are finite numbers 
        const maxFitness = isFinite(data.max_fitness) ? parseFloat(data.max_fitness) : 0;
        const avgFitness = isFinite(data.avg_fitness) ? parseFloat(data.avg_fitness) : 0;
        
        console.log('Adding fitness values to chart - max:', maxFitness, 'avg:', avgFitness);
        
        // Add the generation number to labels
        evolutionChart.data.labels.push(data.generation);
        
        // Add the fitness values to datasets
        evolutionChart.data.datasets[0].data.push(maxFitness);
        evolutionChart.data.datasets[1].data.push(avgFitness);
        
        // Update Y-axis limits if needed based on data
        const maxValue = Math.max(...evolutionChart.data.datasets[0].data);
        console.log('Current max fitness in chart:', maxValue);
        
        // Set appropriate Y-axis max, ensuring small values are still visible
        let yAxisMax;
        if (maxValue <= 0.001) {
            // For very small values, use a small fixed scale
            yAxisMax = 0.01;
        } else if (maxValue < 0.1) {
            // For small values but not tiny, use a scale that makes them visible
            yAxisMax = Math.max(0.1, maxValue * 1.5);
        } else {
            // For normal values, add a 10% margin
            yAxisMax = Math.max(1, Math.ceil(maxValue * 1.1));
        }
        
        console.log('Setting Y-axis max to:', yAxisMax);
        evolutionChart.options.scales.y.max = yAxisMax;
        
        // Update the chart
        evolutionChart.update();
    } catch (error) {
        console.error('Error updating chart with single point:', error);
    }
}

/**
 * Update the evolution chart with full data
 */
function updateEvolutionChartWithFullData(chartData) {
    if (!evolutionChart) {
        console.warn('Chart not initialized yet, initializing now');
        initializeEvolutionChart();
    }
    
    try {
        console.log('Raw chart data received:', JSON.stringify(chartData));
        
        // Clean and validate the data
        const generations = Array.isArray(chartData.generations) ? chartData.generations : [];
        let maxFitness = Array.isArray(chartData.max_fitness) ? chartData.max_fitness : [];
        let avgFitness = Array.isArray(chartData.avg_fitness) ? chartData.avg_fitness : [];
        
        // Ensure both arrays have data
        if (generations.length === 0 || maxFitness.length === 0) {
            console.warn('Empty data received for chart:', chartData);
            return;
        }
        
        // Make sure all arrays are the same length
        const minLength = Math.min(generations.length, maxFitness.length, avgFitness.length);
        if (minLength < generations.length) {
            console.warn(`Mismatched array lengths: gen=${generations.length}, max=${maxFitness.length}, avg=${avgFitness.length}`);
            generations.length = minLength;
            maxFitness.length = minLength;
            avgFitness.length = minLength;
        }
        
        // Replace any infinite or non-numeric values with drawable values
        maxFitness = maxFitness.map(val => isFinite(val) ? parseFloat(val) : 0);
        avgFitness = avgFitness.map(val => isFinite(val) ? parseFloat(val) : 0);
        
        // Log out the values for debugging
        console.log('Cleaned max fitness values:', maxFitness);
        console.log('Cleaned avg fitness values:', avgFitness);
        
        // Find the maximum fitness value to adjust Y-axis
        const maxValue = Math.max(...maxFitness);
        console.log('Max fitness value in data:', maxValue);
        
        // Set appropriate Y-axis max with margin, ensuring small values are still visible
        let yAxisMax;
        if (maxValue <= 0.001) {
            // For very small values, use a small fixed scale
            yAxisMax = 0.01;
        } else if (maxValue < 0.1) {
            // For small values but not tiny, use a scale that makes them visible
            yAxisMax = Math.max(0.1, maxValue * 1.5);
        } else {
            // For normal values, add a 10% margin
            yAxisMax = Math.max(1, Math.ceil(maxValue * 1.1));
        }
        
        console.log('Setting Y-axis max to:', yAxisMax);
        
        // Update chart data
        evolutionChart.data.labels = generations;
        evolutionChart.data.datasets[0].data = maxFitness;
        evolutionChart.data.datasets[1].data = avgFitness;
        
        // Adjust Y-axis limits
        evolutionChart.options.scales.y.max = yAxisMax;
        
        // Update the chart
        evolutionChart.update();
        
        console.log('Chart updated with full data');
    } catch (error) {
        console.error('Error updating chart with full data:', error);
    }
}

/**
 * Update the mutations display
 */
function updateMutationsDisplay(mutations) {
    const container = document.getElementById('mutationsContainer');
    
    if (!container) return;
    
    // Clear the container if it's showing the loading spinner
    if (container.querySelector('.spinner-border')) {
        container.innerHTML = '';
    }
    
    // Create mutation display elements
    mutations.forEach((mutation, index) => {
        const mutationId = `mutation-${mutation.id}`;
        
        // Check if this mutation already exists
        const existingMutation = document.getElementById(mutationId);
        
        if (existingMutation) {
            // Update existing mutation
            const codeElem = existingMutation.querySelector('.mutation-code');
            if (codeElem && codeElem.textContent !== mutation.source) {
                codeElem.textContent = mutation.source;
                existingMutation.classList.add('new-mutation');
                
                // Highlight the code
                if (hljs) {
                    codeElem.innerHTML = hljs.highlight(mutation.source, {language: 'python'}).value;
                }
                
                // Remove the animation class after animation completes
                setTimeout(() => {
                    existingMutation.classList.remove('new-mutation');
                }, 2000);
            }
        } else {
            // Create new mutation element
            const mutationElem = document.createElement('div');
            mutationElem.id = mutationId;
            mutationElem.className = 'mutation-item new-mutation';
            
            // Create header
            const header = document.createElement('div');
            header.className = 'mutation-header';
            header.innerHTML = `
                <span>Variant ${mutation.id + 1}</span>
                <span class="badge bg-secondary">${mutation.name}</span>
            `;
            
            // Create code display
            const codeElem = document.createElement('div');
            codeElem.className = 'mutation-code';
            
            // Highlight the code
            if (hljs) {
                codeElem.innerHTML = hljs.highlight(mutation.source, {language: 'python'}).value;
            } else {
                codeElem.textContent = mutation.source;
            }
            
            // Add elements to the mutation
            mutationElem.appendChild(header);
            mutationElem.appendChild(codeElem);
            
            // Add to container
            container.appendChild(mutationElem);
            
            // Remove the animation class after animation completes
            setTimeout(() => {
                mutationElem.classList.remove('new-mutation');
            }, 2000);
        }
    });
}

/**
 * Fetch the final results
 */
function fetchFinalResults(processId) {
    fetch(`/api/result/${processId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Display the final results
            displayFinalResults(data);
            
            // Explicitly show the final results section in case it wasn't shown before
            document.getElementById('finalResultsSection').style.display = 'block';
            
            // Scroll to the final results section
            document.getElementById('finalResultsSection').scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            showError('Error fetching results: ' + error.message);
        });
}

/**
 * Display the final results
 */
function displayFinalResults(data) {
    // Debug: Log the entire final results data
    console.log('Final results data received:', data);
    
    // Show the final results section
    document.getElementById('finalResultsSection').style.display = 'block';
    
    // Make sure to stop the elapsed time counter
    stopElapsedTimeCounter();
    
    // Clear any intervals to be sure
    if (statusInterval) clearInterval(statusInterval);
    
    // Update the summary
    const resultsSummary = document.getElementById('resultsSummary');
    
    // Check if performance_comparison exists and has the speed_improvement_percentage property
    const performanceComparison = data.performance_comparison || {};
    const improvement = performanceComparison.speed_improvement_percentage !== undefined 
        ? performanceComparison.speed_improvement_percentage 
        : 0;
    
    // Debug: Log performance comparison details
    console.log('Performance comparison data:', performanceComparison);
    console.log('Improvement percentage:', improvement);
    
    let message;
    if (improvement > 0) {
        message = `<i class="bi bi-check-circle me-2"></i>Evolution completed successfully with <strong>${improvement.toFixed(2)}%</strong> performance improvement!`;
        resultsSummary.className = 'alert alert-success';
    } else if (improvement === 0) {
        message = `<i class="bi bi-info-circle me-2"></i>Evolution completed, but no performance improvement was achieved.`;
        resultsSummary.className = 'alert alert-info';
    } else {
        message = `<i class="bi bi-exclamation-triangle me-2"></i>Evolution completed, but performance decreased by <strong>${Math.abs(improvement).toFixed(2)}%</strong>.`;
        resultsSummary.className = 'alert alert-warning';
    }
    
    resultsSummary.innerHTML = message;
    
    // Update metrics with safe defaults
    document.getElementById('performanceImprovement').textContent = `${improvement.toFixed(2)}%`;
    document.getElementById('totalGenerations').textContent = data.generations_completed || 0;
    document.getElementById('totalTime').textContent = formatTime(Math.floor(data.total_time || 0));
    document.getElementById('testCaseCount').textContent = (data.test_cases || []).length;
    
    // Update code displays
    const originalCodeElem = document.getElementById('originalCode');
    const evolvedCodeElem = document.getElementById('evolvedCode');
    
    if (originalCodeElem) {
        originalCodeElem.textContent = data.original_source || 'No original code available';
        hljs.highlightElement(originalCodeElem);
    }
    
    if (evolvedCodeElem) {
        evolvedCodeElem.textContent = data.evolved_source || 'No evolved code available';
        hljs.highlightElement(evolvedCodeElem);
        console.log('Updated evolved code display with:', 
            data.evolved_source 
                ? data.evolved_source.substring(0, 100) + '...' 
                : 'No evolved code available');
        
        // Highlight the evolved code section for visibility
        const evolvedCodeCard = evolvedCodeElem.closest('.card');
        if (evolvedCodeCard) {
            evolvedCodeCard.style.boxShadow = '0 0 15px rgba(25, 135, 84, 0.5)';
            // Add a subtle pulse animation
            evolvedCodeCard.style.animation = 'pulse 2s';
            evolvedCodeCard.style.animationIterationCount = '3';
            evolvedCodeCard.classList.add('evolved-highlight');
            
            // Add a header label to make it clear this is the optimized version
            const cardHeader = evolvedCodeCard.querySelector('.card-header');
            if (cardHeader) {
                cardHeader.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">
                            <span class="badge bg-success me-2">OPTIMIZED</span>
                            Evolved Code
                        </h5>
                        <button id="downloadBtn" class="btn btn-sm btn-outline-primary">
                            <i class="bi bi-download me-1"></i>Download
                        </button>
                    </div>
                `;
            }
        }
    }
    
    // Set the frozen elapsed time on the progress card too
    const elapsedTimeElem = document.getElementById('timeElapsed');
    if (elapsedTimeElem && data.total_time) {
        elapsedTimeElem.textContent = formatTime(Math.floor(data.total_time));
    }
    
    // Update performance table
    updatePerformanceTable(data.performance_comparison);
    
    // Update test cases table
    updateTestCasesTable(data.test_cases);
    
    // Make sure the results section is visible and scrolled into view
    document.getElementById('finalResultsSection').style.display = 'block';
    
    // Add a slight delay before scrolling to ensure DOM is updated
    setTimeout(() => {
        // Get the evolved code card and scroll to it for better visibility
        const evolvedCodeCard = document.querySelector('#evolvedCode').closest('.card');
        if (evolvedCodeCard) {
            evolvedCodeCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            // Fall back to scrolling to the results section
            document.getElementById('finalResultsSection').scrollIntoView({ behavior: 'smooth' });
        }
        
        // Make sure the download button works after replacing the header
        const downloadBtn = document.getElementById('downloadBtn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', downloadEvolvedCode);
        }
    }, 300);
}

/**
 * Update the performance comparison table
 */
function updatePerformanceTable(comparison) {
    const table = document.getElementById('performanceTable');
    if (!table) return;
    
    // Handle null or undefined comparison
    comparison = comparison || {};
    
    // Debug: Log the performance comparison data
    console.log('Updating performance table with data:', comparison);
    
    // Clear existing rows
    table.innerHTML = '';
    
    // Add speed row
    const speedRow = document.createElement('tr');
    
    // Speed values are now fitness scores (0-10) where higher is better
    const originalSpeed = comparison.original_speed !== undefined ? comparison.original_speed : 0;
    const evolvedSpeed = comparison.evolved_speed !== undefined ? comparison.evolved_speed : 0;
    const speedImprovement = comparison.speed_improvement_percentage !== undefined ? comparison.speed_improvement_percentage : 0;
    
    // Debug: Log specific values after parsing
    console.log('Original speed:', originalSpeed);
    console.log('Evolved speed:', evolvedSpeed);
    console.log('Speed improvement:', speedImprovement);
    
    speedRow.innerHTML = `
        <td>Speed (fitness)</td>
        <td>${originalSpeed.toFixed(2)}</td>
        <td>${evolvedSpeed.toFixed(2)}</td>
        <td class="${speedImprovement > 0 ? 'metric-improved' : 'metric-worsened'}">${speedImprovement.toFixed(2)}%</td>
    `;
    
    // Add accuracy row
    const accuracyRow = document.createElement('tr');
    
    const originalAccuracy = comparison.original_accuracy !== undefined ? comparison.original_accuracy : 0;
    const evolvedAccuracy = comparison.evolved_accuracy !== undefined ? comparison.evolved_accuracy : 0;
    const accuracyImprovement = comparison.accuracy_improvement !== undefined ? comparison.accuracy_improvement : 0;
    
    // Debug: Log accuracy values
    console.log('Original accuracy:', originalAccuracy);
    console.log('Evolved accuracy:', evolvedAccuracy);
    console.log('Accuracy improvement:', accuracyImprovement);
    
    accuracyRow.innerHTML = `
        <td>Accuracy</td>
        <td>${(originalAccuracy * 100).toFixed(2)}%</td>
        <td>${(evolvedAccuracy * 100).toFixed(2)}%</td>
        <td class="${accuracyImprovement >= 0 ? 'metric-improved' : 'metric-worsened'}">${(accuracyImprovement * 100).toFixed(2)}%</td>
    `;
    
    // Add rows to table
    table.appendChild(speedRow);
    table.appendChild(accuracyRow);
}

/**
 * Update the test cases table
 */
function updateTestCasesTable(testCases) {
    const table = document.getElementById('testCasesTable');
    if (!table) return;
    
    // Handle null or undefined testCases
    testCases = testCases || [];
    
    // Clear existing rows
    table.innerHTML = '';
    
    // If no test cases, show a message
    if (testCases.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="3" class="text-center">No test cases available</td>
        `;
        table.appendChild(row);
        return;
    }
    
    // Add rows for each test case
    testCases.forEach(testCase => {
        const row = document.createElement('tr');
        
        row.innerHTML = `
            <td>${testCase.id || 'N/A'}</td>
            <td><code>${testCase.args || 'N/A'}</code></td>
            <td><code>${testCase.expected || 'N/A'}</code></td>
        `;
        
        table.appendChild(row);
    });
}

/**
 * Cancel the evolution process
 */
function cancelEvolution() {
    if (!processId) return;
    
    // Update status
    updateStatus('Cancelling evolution process...', 'warning');
    
    // Send cancellation request
    fetch(`/api/cancel/${processId}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        updateStatus(data.message, 'warning');
        
        // Clear intervals
        if (statusInterval) {
            clearInterval(statusInterval);
        }
        stopElapsedTimeCounter();
    })
    .catch(error => {
        showError('Error cancelling evolution: ' + error.message);
    });
}

/**
 * Restart the evolution process
 */
function restartEvolution() {
    // Hide the final results section
    document.getElementById('finalResultsSection').style.display = 'none';
    
    // Hide the results section
    document.getElementById('resultsSection').style.display = 'none';
    
    // Clear any intervals
    clearInterval(statusInterval);
    clearInterval(elapsedTimeInterval);
    
    // Reset process ID
    processId = null;
    
    // Destroy the chart
    if (evolutionChart) {
        evolutionChart.destroy();
        evolutionChart = null;
    }
    
    // Reset form
    document.getElementById('codeForm').reset();
    
    // Focus on the code editor
    if (editor) {
        editor.setValue('');
        editor.focus();
    }
}

/**
 * Download the evolved code
 */
function downloadEvolvedCode() {
    const evolvedCode = document.getElementById('evolvedCode');
    
    if (!evolvedCode || !evolvedCode.textContent) {
        showError('No evolved code to download.');
        return;
    }
    
    // Create a blob with the code
    const blob = new Blob([evolvedCode.textContent], {type: 'text/plain'});
    
    // Create a download link
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `evolved_${document.getElementById('functionName').value || 'function'}.py`;
    
    // Trigger download
    document.body.appendChild(a);
    a.click();
    
    // Cleanup
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Update performance metrics display
 */
function updatePerformanceDisplay(performance) {
    try {
        const improvement = performance.speed_improvement || 0;
        
        // Update the improvement display
        if (document.getElementById('performanceImprovement')) {
            document.getElementById('performanceImprovement').textContent = `${improvement.toFixed(2)}%`;
        }
        
        // Update the summary based on improvement
        const resultsSummary = document.getElementById('resultsSummary');
        if (resultsSummary) {
            let message;
            if (improvement > 0) {
                message = `<i class="bi bi-check-circle me-2"></i>Evolution completed successfully with <strong>${improvement.toFixed(2)}%</strong> performance improvement!`;
                resultsSummary.className = 'alert alert-success';
            } else if (improvement === 0) {
                message = `<i class="bi bi-info-circle me-2"></i>Evolution completed, but no performance improvement was achieved.`;
                resultsSummary.className = 'alert alert-info';
            } else {
                message = `<i class="bi bi-exclamation-triangle me-2"></i>Evolution completed, but performance decreased by <strong>${Math.abs(improvement).toFixed(2)}%</strong>.`;
                resultsSummary.className = 'alert alert-warning';
            }
            resultsSummary.innerHTML = message;
        }
        
        // Update the performance table if it exists
        updatePerformanceTable(performance);
    } catch (error) {
        console.error('Error updating performance display:', error);
    }
} 