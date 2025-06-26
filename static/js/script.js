// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupImageUpload();
    setupBMICalculation();
    setupFormValidation();
    setupAnimations();
}

// Image Upload Functionality
function setupImageUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('image');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');

    if (!uploadArea || !fileInput) return;

    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File selection
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    function handleFileUpload(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            showAlert('Please select a valid image file.', 'error');
            return;
        }

        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            showAlert('File size should be less than 10MB.', 'error');
            return;
        }

        // Preview image
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            imagePreview.style.display = 'block';
            uploadArea.style.display = 'none';
        };
        reader.readAsDataURL(file);

        // Update upload area text
        uploadArea.querySelector('.upload-content').innerHTML = `
            <i class="fas fa-check-circle fa-3x text-success mb-3"></i>
            <p class="text-success">Image uploaded successfully!</p>
            <p class="text-muted small">Click to change image</p>
        `;
    }
}

// BMI Calculation
function setupBMICalculation() {
    const heightFeet = document.getElementById('height_feet');
    const weightKg = document.getElementById('weight_kg');
    const bmiDisplay = document.getElementById('bmiDisplay');
    const bmiValue = document.getElementById('bmiValue');
    const bmiCategory = document.getElementById('bmiCategory');

    if (!heightFeet || !weightKg) return;

    const inputs = [heightFeet, weightKg];
    
    inputs.forEach(input => {
        input.addEventListener('input', calculateBMI);
    });

    function calculateBMI() {
        const feet = parseFloat(heightFeet.value) || 0;
        const weight = parseFloat(weightKg.value) || 0;

        if (feet > 0 && weight > 0) {
            // Convert decimal feet to feet and inches
            const feetInt = Math.floor(feet);
            const inches = Math.round((feet - feetInt) * 12);
            const totalHeightInches = (feetInt * 12) + inches;
            const heightMeters = totalHeightInches * 0.0254;
            const bmi = weight / (heightMeters ** 2);

            if (!isNaN(bmi) && isFinite(bmi)) {
                bmiValue.textContent = bmi.toFixed(2);
                bmiCategory.textContent = getBMICategory(bmi);
                bmiDisplay.style.display = 'block';
                
                // Add animation
                bmiDisplay.classList.add('animate__animated', 'animate__fadeIn');
            }
        } else {
            bmiDisplay.style.display = 'none';
        }
    }

    function getBMICategory(bmi) {
        if (bmi < 18.5) return 'Underweight';
        if (bmi < 25) return 'Normal';
        if (bmi < 30) return 'Overweight';
        return 'Obese';
    }
}

// Form Validation
function setupFormValidation() {
    const form = document.getElementById('fitnessForm');
    
    if (!form) return;

    form.addEventListener('submit', function(e) {
        if (!validateForm()) {
            e.preventDefault();
        }
    });

    function validateForm() {
        const requiredFields = form.querySelectorAll('[required]');
        let isValid = true;

        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                showFieldError(field, 'This field is required.');
                isValid = false;
            } else {
                clearFieldError(field);
            }
        });

        // Validate age
        const age = document.getElementById('age');
        if (age && age.value) {
            const ageNum = parseInt(age.value);
            if (ageNum < 13 || ageNum > 100) {
                showFieldError(age, 'Age must be between 13 and 100.');
                isValid = false;
            }
        }

        // Validate height
        const heightFeet = document.getElementById('height_feet');
        if (heightFeet && heightFeet.value) {
            const feet = parseFloat(heightFeet.value);
            if (feet < 3 || feet > 8) {
                showFieldError(heightFeet, 'Please enter a valid height.');
                isValid = false;
            }
        }

        // Validate weight
        const weight = document.getElementById('weight_kg');
        if (weight && weight.value) {
            const weightNum = parseFloat(weight.value);
            if (weightNum < 30 || weightNum > 300) {
                showFieldError(weight, 'Weight must be between 30 and 300 kg.');
                isValid = false;
            }
        }

        // Validate image upload
        const imageInput = document.getElementById('image');
        if (imageInput && !imageInput.files[0]) {
            showAlert('Please upload a full body image.', 'error');
            isValid = false;
        }

        return isValid;
    }

    function showFieldError(field, message) {
        clearFieldError(field);
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'invalid-feedback d-block';
        errorDiv.textContent = message;
        
        field.classList.add('is-invalid');
        field.parentNode.appendChild(errorDiv);
    }

    function clearFieldError(field) {
        field.classList.remove('is-invalid');
        const errorDiv = field.parentNode.querySelector('.invalid-feedback');
        if (errorDiv) {
            errorDiv.remove();
        }
    }
}

// Animations
function setupAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate__animated', 'animate__fadeInUp');
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animateElements = document.querySelectorAll('.feature-card, .meal-card, .workout-card, .recommendation-item');
    animateElements.forEach(el => {
        observer.observe(el);
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Alert System
function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => alert.remove());

    // Create new alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Insert alert at the top of the form
    const form = document.getElementById('fitnessForm');
    if (form) {
        form.parentNode.insertBefore(alertDiv, form);
    } else {
        document.body.insertBefore(alertDiv, document.body.firstChild);
    }

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Utility Functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

// Loading State Management
function showLoading(button) {
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="loading"></span> Processing...';
    button.disabled = true;
    return originalText;
}

function hideLoading(button, originalText) {
    button.innerHTML = originalText;
    button.disabled = false;
}

// Form Submission Enhancement
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('fitnessForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            const submitButton = form.querySelector('button[type="submit"]');
            const originalText = showLoading(submitButton);
            
            // Simulate processing time (remove in production)
            setTimeout(() => {
                hideLoading(submitButton, originalText);
            }, 2000);
        });
    }
});

// Responsive Navigation
function setupResponsiveNav() {
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');
    
    if (navbarToggler && navbarCollapse) {
        navbarToggler.addEventListener('click', function() {
            navbarCollapse.classList.toggle('show');
        });
    }
}

// Initialize responsive navigation
document.addEventListener('DOMContentLoaded', setupResponsiveNav);

// Print Functionality
function printPlan() {
    window.print();
}

// Download Functionality (placeholder)
function downloadPlan() {
    // This would integrate with a PDF generation library
    showAlert('PDF download feature will be implemented soon!', 'info');
}

// Export functions for global access
window.printPlan = printPlan;
window.downloadPlan = downloadPlan; 