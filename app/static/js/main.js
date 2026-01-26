// Telecom Churn Prediction - Main JavaScript
// Enhanced with modern animations, notifications & smooth interactions

// ========================================================================
// UTILITY FUNCTIONS - USING NEW NOTIFICATION SYSTEM
// ========================================================================

function showAlert(message, type = 'info') {
    // Map Bootstrap alert types to Toast types
    const typeMap = {
        'success': 'success',
        'danger': 'error',
        'warning': 'warning',
        'info': 'info'
    };
    
    const toastType = typeMap[type] || 'info';
    
    // Use new toast system if available
    if (window.toast) {
        window.toast.show(message, toastType, 4000);
    }
}

function showLoading(element, message = 'Processing...') {
    element.disabled = true;
    element.classList.add('disabled', 'opacity-75');
    element.innerHTML = `<span class="spin inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"></span>${message}`;
}

function hideLoading(element, originalText) {
    element.disabled = false;
    element.classList.remove('disabled', 'opacity-75');
    element.innerHTML = originalText;
}

function animateValue(element, start, end, duration = 1000) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= end) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current * 100) / 100 + '%';
    }, 16);
}

// ========================================================================
// FORM VALIDATION WITH REAL-TIME FEEDBACK
// ========================================================================

function setupFormValidation(form) {
    const inputs = form.querySelectorAll('input, select, textarea');
    
    inputs.forEach(input => {
        input.addEventListener('change', () => {
            validateField(input);
        });
        
        input.addEventListener('blur', () => {
            validateField(input);
        });
    });
}

function validateField(field) {
    const parent = field.parentElement;
    
    if (!field.value) {
        parent.classList.remove('is-valid');
        parent.classList.add('is-invalid');
        return false;
    } else {
        parent.classList.remove('is-invalid');
        parent.classList.add('is-valid');
        return true;
    }
}

// ========================================================================
// PREDICTION FORM HANDLER - ENHANCED
// ========================================================================

async function handlePrediction(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalBtnText = submitBtn.innerHTML;
    
    // Start loading animation
    animator.startLoading();
    
    try {
        // Validate all fields
        const inputs = form.querySelectorAll('input, select, textarea');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!validateField(input)) {
                isValid = false;
            }
        });
        
        if (!isValid) {
            showAlert('Please fill in all required fields', 'warning');
            animator.completeLoading();
            return;
        }
        
        showLoading(submitBtn, 'Getting Prediction...');
        
        // Collect form data
        const formData = new FormData(form);
        const data = {};
        
        formData.forEach((value, key) => {
            const numValue = parseFloat(value);
            data[key] = isNaN(numValue) ? value : numValue;
        });
        
        console.log('🚀 Sending prediction data:', data);
        
        // Send prediction request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify(data),
            timeout: 30000
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || result.message || 'Prediction failed. Please run the ML pipeline first.');
        }
        
        displayResults(result);
        showAlert('✨ Prediction completed successfully!', 'success');
        animator.completeLoading();
        
    } catch (error) {
        console.error('❌ Prediction error:', error);
        showAlert(`Prediction Failed: ${error.message}`, 'danger');
        animator.completeLoading();
        
        // Show detailed error message
        if (error.message.includes('Model not trained')) {
            showAlert('⚠️ Models not trained yet. Run: python main.py', 'warning');
        }
    } finally {
        hideLoading(submitBtn, originalBtnText);
    }
}

// ========================================================================
// RESULTS DISPLAY - ENHANCED WITH ANIMATIONS
// ========================================================================

function displayResults(result) {
    console.log('📊 displayResults called with:', result);
    
    const resultsSection = document.getElementById('results-section');
    
    if (!resultsSection) {
        console.error('❌ results-section element not found!');
        return; // Not on prediction page
    }
    
    console.log('✓ Found results-section element');
    
    // Show results section with animation
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
    
    // Animate probability meter
    const probability = result.churn_probability * 100;
    const probabilityFill = document.getElementById('probability-fill');
    
    if (!probabilityFill) {
        console.error('❌ probability-fill element not found!');
    } else {
        console.log('✓ Found probability-fill element');
        probabilityFill.style.width = '0%';
        
        // Animate the fill
        setTimeout(() => {
            probabilityFill.style.transition = 'width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)';
            probabilityFill.style.width = probability + '%';
            probabilityFill.textContent = probability.toFixed(2) + '%';
            console.log(`✓ Updated probability to ${probability.toFixed(2)}%`);
        }, 100);
    }
    
    // Update risk category badge
    const riskBadge = document.getElementById('risk-badge');
    if (!riskBadge) {
        console.error('❌ risk-badge element not found!');
    } else {
        console.log('✓ Found risk-badge element');
        riskBadge.className = `risk-badge risk-${result.risk_category.toLowerCase()}`;
        riskBadge.innerHTML = `<strong>${result.risk_category} RISK</strong>`;
        riskBadge.style.animation = 'slideInRight 0.5s ease-out';
    }
    
    // Update suggestion with better formatting
    const suggestionBox = document.getElementById('suggestion');
    if (!suggestionBox) {
        console.error('❌ suggestion element not found!');
    } else {
        console.log('✓ Found suggestion element');
        suggestionBox.classList.remove('suggestion-box');
        void suggestionBox.offsetWidth; // Trigger reflow
        suggestionBox.classList.add('suggestion-box');
        suggestionBox.innerHTML = `
            <strong style="color: #1a73e8; font-size: 1.1rem;">Recommendation:</strong><br/>
            ${result.suggestion}
        `;
    }
    
    // Update model accuracy if available
    if (result.model_accuracy) {
        const accuracyBox = document.getElementById('model-accuracy');
        if (accuracyBox) {
            accuracyBox.textContent = result.model_accuracy;
        }
    }
    
    // Store result for reference
    sessionStorage.setItem('lastResult', JSON.stringify(result));
    
    // Scroll to results smoothly
    setTimeout(() => {
        resultsSection.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }, 500);
    
    console.log('✓ Results displayed successfully');
}

// ========================================================================
// FORM RESET
// ========================================================================

function resetForm() {
    const form = document.querySelector('form');
    if (!form) return;
    
    if (confirm('Reset all fields and results?')) {
        form.reset();
        form.classList.remove('was-validated');
        
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
        
        showAlert('Form reset successfully', 'success');
    }
}

// ========================================================================
// EXAMPLE DATA LOADER
// ========================================================================

function loadExampleData() {
    const exampleData = {
        customerID: '0002-ORFBO',
        tenure: '24',
        MonthlyCharges: '65.50',
        TotalCharges: '1570.00',
        gender: 'Male',
        SeniorCitizen: '0',
        Partner: 'Yes',
        Dependents: 'No',
        PhoneService: 'Yes',
        MultipleLines: 'No',
        InternetService: 'Fiber optic',
        OnlineSecurity: 'No',
        OnlineBackup: 'Yes',
        DeviceProtection: 'No',
        TechSupport: 'No',
        StreamingTV: 'No',
        StreamingMovies: 'No',
        Contract: 'Two year',
        PaymentMethod: 'Electronic check',
        PaperlessBilling: 'Yes'
    };
    
    Object.entries(exampleData).forEach(([key, value]) => {
        const field = document.querySelector(`[name="${key}"]`);
        if (field) {
            field.value = value;
        }
    });
    
    showAlert('Example data loaded! Click "Predict Churn" to test.', 'info');
}

// ========================================================================
// FORM VALIDATION
// ========================================================================

function validateForm(form) {
    const fields = form.querySelectorAll('[required]');
    let isValid = true;
    
    fields.forEach(field => {
        if (!field.value.trim()) {
            field.classList.add('is-invalid');
            isValid = false;
        } else {
            field.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

// ========================================================================
// TOOLTIPS & POPOVERS INITIALIZATION
// ========================================================================

function initTooltips() {
    // Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// ========================================================================
// PAGE INITIALIZATION
// ========================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    initTooltips();
    
    // Add form submission handler
    const predictionForm = document.querySelector('form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePrediction);
    }
    
    // Restore last result if on prediction page
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        const lastResult = sessionStorage.getItem('lastResult');
        if (lastResult) {
            const result = JSON.parse(lastResult);
            displayResults(result);
        }
    }
    
    // Add smooth scrolling to all links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    console.log('Telecom Churn Prediction App Initialized');
});

// ========================================================================
// WINDOW ERROR HANDLER
// ========================================================================

window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showAlert('An unexpected error occurred. Please refresh the page.', 'danger');
});
