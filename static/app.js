// API endpoint
const API_URL = window.location.origin;

// DOM elements
const textInput = document.getElementById('text-input');
const charCount = document.getElementById('char-count');
const analyzeBtn = document.getElementById('analyze-btn');
const loading = document.getElementById('loading');
const result = document.getElementById('result');
const error = document.getElementById('error');
const resultContent = document.getElementById('result-content');
const errorMessage = document.getElementById('error-message');

// Update character count
textInput.addEventListener('input', () => {
    const count = textInput.value.length;
    charCount.textContent = `${count} characters`;
    
    // Enable/disable button based on minimum length
    analyzeBtn.disabled = count < 10;
});

// Handle analyze button click
analyzeBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    
    if (text.length < 10) {
        showError('Text must be at least 10 characters long');
        return;
    }
    
    await analyzeText(text);
});

// Handle Enter key (Ctrl+Enter to submit)
textInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        analyzeBtn.click();
    }
});

async function analyzeText(text) {
    // Hide previous results/errors
    hideAll();
    showLoading();
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const responseData = await response.json();
        
        // Handle new standardized response format
        if (!response.ok || !responseData.success) {
            const errorMsg = responseData.error || responseData.message || 'Failed to analyze text';
            throw new Error(errorMsg);
        }
        
        // Extract data from the standardized response
        const data = responseData.data || responseData;
        displayResult(data);
        
    } catch (err) {
        showError(err.message || 'An error occurred while analyzing the text');
    } finally {
        hideLoading();
    }
}

function displayResult(data) {
    // Handle both old and new response formats for backward compatibility
    const isAI = data.is_ai;
    const label = data.label;
    const confidence = (data.confidence * 100).toFixed(1);
    
    // Safely access probabilities with fallback
    const probabilities = data.probabilities || {};
    const humanProb = probabilities.human ? (probabilities.human * 100).toFixed(2) : '0.00';
    const aiProb = probabilities.ai_generated ? (probabilities.ai_generated * 100).toFixed(2) : '0.00';
    
    resultContent.innerHTML = `
        <div class="result-section">
            <div class="result-metric">
                <div class="metric-label">Classification</div>
                <div class="metric-value ${isAI ? 'ai' : 'human'}">${label}</div>
                <div class="metric-description">
                    <span class="classification-badge ${isAI ? 'ai' : 'human'}">${isAI ? 'AI-Generated' : 'Human-Written'}</span>
                </div>
            </div>
            
            <div class="result-metric">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">${confidence}%</div>
                <div class="metric-description">Model confidence score</div>
            </div>
        </div>
        
        <div class="confidence-bar-container">
            <div class="confidence-bar-label">
                <span>Confidence Level</span>
                <span>${confidence}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-bar-fill ${isAI ? 'ai' : 'human'}" 
                     style="width: ${confidence}%"></div>
            </div>
        </div>
        
        <div>
            <table class="probability-table">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Probability</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Human-Written</td>
                        <td class="probability-value">${humanProb}%</td>
                    </tr>
                    <tr>
                        <td>AI-Generated</td>
                        <td class="probability-value">${aiProb}%</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;
    
    showResult();
}

function showLoading() {
    loading.classList.remove('hidden');
}

function hideLoading() {
    loading.classList.add('hidden');
}

function showResult() {
    result.classList.remove('hidden');
    error.classList.add('hidden');
}

function showError(message) {
    errorMessage.textContent = message;
    error.classList.remove('hidden');
    result.classList.add('hidden');
}

function hideAll() {
    result.classList.add('hidden');
    error.classList.add('hidden');
    loading.classList.add('hidden');
}

// Check API health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) {
            console.warn('API health check failed');
        }
    } catch (err) {
        console.warn('Could not connect to API:', err);
    }
});
