/**
 * Facial Recognition Attendance System - Complete Working Script
 */

// Global variables
let videoStream = null;
let videoElement = null;
let canvasElement = null;
let isProcessing = false;
let autoRecognitionInterval = null;
let lastRecognitionTime = {};

// DOM Elements
let currentPage = window.location.pathname.split('/').pop() || 'dashboard';

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("Face Recognition System Initializing...");
    
    // Initialize elements
    videoElement = document.getElementById('video');
    canvasElement = document.getElementById('canvas');
    
    // Update date and time
    updateDateTime();
    setInterval(updateDateTime, 1000);
    
    // Setup page-specific functionality
    setupPage();
    
    // Test system connection
    testSystem();
    
    // Update system stats every 30 seconds
    setInterval(updateSystemStats, 30000);
});

// Setup page-specific functionality
function setupPage() {
    switch(currentPage) {
        case 'dashboard':
            setupDashboard();
            break;
        case 'recognize':
            setupRecognitionPage();
            break;
        case 'register':
            setupRegistrationPage();
            break;
        case 'attendance':
            setupAttendancePage();
            break;
        case 'report':
            setupReportPage();
            break;
        case 'students':
            setupStudentsPage();
            break;
    }
}

// ===================
// COMMON FUNCTIONS
// ===================

function updateDateTime() {
    const now = new Date();
    const dateElement = document.getElementById('current-date');
    const timeElement = document.getElementById('current-time');
    
    if (dateElement) {
        dateElement.textContent = now.toLocaleDateString('en-US', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
    }
    
    if (timeElement) {
        timeElement.textContent = now.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit',
            hour12: true 
        });
    }
}

async function testSystem() {
    try {
        const response = await fetch('/api/face-test');
        if (response.ok) {
            const data = await response.json();
            console.log("System Status:", data);
            showNotification(`System ready with ${data.embeddings_count} registered faces`, 'success');
        }
    } catch (error) {
        console.error("System test failed:", error);
        showNotification('System test failed. Please check server connection.', 'error');
    }
}

function updateSystemStats() {
    if (currentPage === 'dashboard') {
        loadDashboardStats();
    }
}

function showNotification(message, type = 'info', duration = 5000) {
    // Remove existing notifications
    const existing = document.querySelectorAll('.notification');
    existing.forEach(n => n.remove());
    
    // Create notification
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    let icon = 'fa-info-circle';
    switch(type) {
        case 'success': icon = 'fa-check-circle'; break;
        case 'error': icon = 'fa-exclamation-circle'; break;
        case 'warning': icon = 'fa-exclamation-triangle'; break;
    }
    
    notification.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">&times;</button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after duration
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, duration);
    
    // Add click to dismiss
    notification.addEventListener('click', (e) => {
        if (e.target !== notification.querySelector('button')) {
            notification.remove();
        }
    });
}

// ===================
// DASHBOARD PAGE
// ===================

function setupDashboard() {
    loadDashboardStats();
    updateLastUpdated();
    
    // Setup quick action buttons
    const exportBtn = document.querySelector('.export-data');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportTodayAttendance);
    }
}

async function loadDashboardStats() {
    try {
        const response = await fetch('/api/attendance/stats');
        if (response.ok) {
            const stats = await response.json();
            
            // Update stats cards
            document.querySelector('.total-students h2').textContent = stats.total_students;
            document.querySelector('.present-today h2').textContent = stats.present_today;
            document.querySelector('.absent-today h2').textContent = stats.absent_today;
            
            // Update recent attendance
            loadRecentAttendance();
        }
    } catch (error) {
        console.error('Failed to load dashboard stats:', error);
    }
}

async function loadRecentAttendance() {
    try {
        const response = await fetch('/api/attendance/today');
        if (response.ok) {
            const attendance = await response.json();
            const tableBody = document.querySelector('.recent-attendance tbody');
            
            if (tableBody) {
                tableBody.innerHTML = '';
                
                if (attendance.length === 0) {
                    tableBody.innerHTML = `
                        <tr>
                            <td colspan="4" class="text-center" style="padding: 20px; color: #6c757d;">
                                No attendance recorded today
                            </td>
                        </tr>
                    `;
                    return;
                }
                
                attendance.slice(0, 5).forEach(record => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${record.student_id}</td>
                        <td>${record.name}</td>
                        <td>${record.time}</td>
                        <td><span class="status-badge status-present">Present</span></td>
                    `;
                    tableBody.appendChild(row);
                });
            }
        }
    } catch (error) {
        console.error('Failed to load recent attendance:', error);
    }
}

function updateLastUpdated() {
    const lastUpdated = document.getElementById('last-updated');
    if (lastUpdated) {
        lastUpdated.textContent = new Date().toLocaleTimeString();
    }
}

// ===================
// RECOGNITION PAGE
// ===================

function setupRecognitionPage() {
    const startBtn = document.getElementById('start-camera');
    const stopBtn = document.getElementById('stop-camera');
    const captureBtn = document.getElementById('capture-face');
    
    if (startBtn) startBtn.addEventListener('click', startCamera);
    if (stopBtn) stopBtn.addEventListener('click', stopCamera);
    if (captureBtn) captureBtn.addEventListener('click', () => captureAndRecognize(false));
    
    // Load attendance log
    loadAttendanceLog();
    
    // Update status
    updateRecognitionStatus('Ready to start. Click "Start Camera" to begin.', 'waiting');
}

async function startCamera() {
    try {
        updateRecognitionStatus('Initializing camera...', 'processing');
        
        // Request camera access
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        
        // Set video source
        videoElement.srcObject = videoStream;
        
        // Enable/disable buttons
        document.getElementById('start-camera').disabled = true;
        document.getElementById('stop-camera').disabled = false;
        document.getElementById('capture-face').disabled = false;
        
        updateRecognitionStatus('Camera started. Look at the camera with good lighting.', 'success');
        
        // Start auto recognition
        startAutoRecognition();
        
    } catch (error) {
        console.error('Camera error:', error);
        updateRecognitionStatus('Camera access failed. Please allow camera permissions.', 'error');
        showNotification('Camera access denied. Please allow camera in browser settings.', 'error');
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
        videoElement.srcObject = null;
    }
    
    document.getElementById('start-camera').disabled = false;
    document.getElementById('stop-camera').disabled = true;
    document.getElementById('capture-face').disabled = true;
    
    // Stop auto recognition
    stopAutoRecognition();
    
    updateRecognitionStatus('Camera stopped.', 'waiting');
}

function startAutoRecognition() {
    if (autoRecognitionInterval) {
        clearInterval(autoRecognitionInterval);
    }
    
    autoRecognitionInterval = setInterval(() => {
        if (videoStream && !isProcessing) {
            captureAndRecognize(true);
        }
    }, 3000); // Recognize every 3 seconds
}

function stopAutoRecognition() {
    if (autoRecognitionInterval) {
        clearInterval(autoRecognitionInterval);
        autoRecognitionInterval = null;
    }
}

async function captureAndRecognize(auto = false) {
    if (!videoStream || isProcessing) return;
    
    isProcessing = true;
    updateRecognitionStatus('Capturing face...', 'processing');
    
    try {
        // Setup canvas
        const canvas = canvasElement || document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        
        // Draw video frame (mirrored)
        context.translate(canvas.width, 0);
        context.scale(-1, 1);
        context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Convert to blob
        canvas.toBlob(async (blob) => {
            if (!blob) {
                updateRecognitionStatus('Failed to capture image', 'error');
                isProcessing = false;
                return;
            }
            
            updateRecognitionStatus('Analyzing face...', 'processing');
            
            try {
                const formData = new FormData();
                formData.append('file', blob, 'face.jpg');
                
                const response = await fetch('/api/recognize', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                console.log("Recognition result:", result);
                
                if (result.recognized && result.student) {
                    // Check if recently recognized (within 30 seconds)
                    const studentId = result.student.student_id;
                    const now = Date.now();
                    
                    if (!lastRecognitionTime[studentId] || (now - lastRecognitionTime[studentId] > 30000)) {
                        // Success recognition
                        lastRecognitionTime[studentId] = now;
                        
                        updateRecognitionStatus(`Welcome ${result.student.name}!`, 'success');
                        
                        // Add to log
                        addToLog(result.student, auto);
                        
                        // Show notification
                        showNotification(`Attendance marked for ${result.student.name}`, 'success');
                        
                        // Show success effect
                        showSuccessEffect();
                        
                        // Refresh attendance log
                        loadAttendanceLog();
                        
                    } else {
                        updateRecognitionStatus(`${result.student.name} already recognized recently`, 'warning');
                        showNotification(`${result.student.name} already marked attendance recently`, 'warning');
                    }
                    
                } else {
                    // Not recognized
                    updateRecognitionStatus('Face not recognized', 'error');
                    if (result.message) {
                        showNotification(result.message, 'warning');
                    }
                }
                
            } catch (error) {
                console.error('Recognition error:', error);
                updateRecognitionStatus('Recognition failed', 'error');
                showNotification('Failed to recognize face. Please try again.', 'error');
            } finally {
                isProcessing = false;
            }
            
        }, 'image/jpeg', 0.9);
        
    } catch (error) {
        console.error('Capture error:', error);
        updateRecognitionStatus('Capture failed', 'error');
        isProcessing = false;
    }
}

function updateRecognitionStatus(message, type) {
    const statusElement = document.getElementById('status-indicator');
    if (!statusElement) return;
    
    // Clear existing classes
    statusElement.className = '';
    
    // Add base and type classes
    statusElement.classList.add('recognition-status');
    
    let icon = 'fa-circle';
    let statusClass = '';
    
    switch(type) {
        case 'processing':
            icon = 'fas fa-sync fa-spin';
            statusClass = 'status-processing';
            break;
        case 'success':
            icon = 'fas fa-check-circle';
            statusClass = 'status-success';
            break;
        case 'error':
            icon = 'fas fa-exclamation-circle';
            statusClass = 'status-error';
            break;
        case 'waiting':
            icon = 'fas fa-circle';
            statusClass = 'status-waiting';
            break;
        case 'warning':
            icon = 'fas fa-exclamation-triangle';
            statusClass = 'status-warning';
            break;
    }
    
    statusElement.classList.add(statusClass);
    statusElement.innerHTML = `<i class="${icon}"></i><span>${message}</span>`;
}

function showSuccessEffect() {
    const overlay = document.getElementById('overlay');
    if (overlay) {
        overlay.style.animation = 'successPulse 1s';
        setTimeout(() => {
            overlay.style.animation = '';
        }, 1000);
    }
}

function addToLog(student, auto = false) {
    const logList = document.getElementById('attendance-log-list');
    if (!logList) return;
    
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit'
    });
    
    const logItem = document.createElement('div');
    logItem.className = 'log-item';
    logItem.innerHTML = `
        <div class="log-info">
            <strong>${student.name}</strong>
            <small>${student.student_id}</small>
        </div>
        <div class="log-details">
            <span class="log-time">${timeString}</span>
            <span class="log-method">${auto ? 'Auto' : 'Manual'}</span>
        </div>
    `;
    
    // Add to top
    if (logList.firstChild) {
        logList.insertBefore(logItem, logList.firstChild);
    } else {
        logList.appendChild(logItem);
    }
    
    // Remove "no records" message
    const noRecords = logList.querySelector('.no-records');
    if (noRecords) noRecords.remove();
    
    // Limit to 10 items
    while (logList.children.length > 10) {
        logList.removeChild(logList.lastChild);
    }
}

async function loadAttendanceLog() {
    try {
        const response = await fetch('/api/attendance/today');
        if (response.ok) {
            const attendance = await response.json();
            const logList = document.getElementById('attendance-log-list');
            
            if (logList) {
                logList.innerHTML = '';
                
                if (attendance.length === 0) {
                    logList.innerHTML = '<div class="no-records">No attendance records for today yet</div>';
                    return;
                }
                
                attendance.forEach(record => {
                    const logItem = document.createElement('div');
                    logItem.className = 'log-item';
                    logItem.innerHTML = `
                        <div class="log-info">
                            <strong>${record.name}</strong>
                            <small>${record.student_id}</small>
                        </div>
                        <div class="log-details">
                            <span class="log-time">${record.time}</span>
                            <span class="log-method">${record.method || 'Face Recognition'}</span>
                        </div>
                    `;
                    logList.appendChild(logItem);
                });
            }
        }
    } catch (error) {
        console.error('Error loading attendance log:', error);
    }
}

// Manual attendance functions
function markManualAttendance() {
    const modal = document.getElementById('manual-attendance-modal');
    if (modal) {
        modal.style.display = 'flex';
    }
}

function closeManualModal() {
    const modal = document.getElementById('manual-attendance-modal');
    if (modal) {
        modal.style.display = 'none';
        const form = document.getElementById('manual-attendance-form');
        if (form) form.reset();
    }
}

async function handleManualAttendance(event) {
    event.preventDefault();
    
    const studentId = document.getElementById('manual-student-id').value.trim();
    const studentName = document.getElementById('manual-name').value.trim();
    
    if (!studentId || !studentName) {
        showNotification('Please fill all fields', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/attendance/mark', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                student_id: studentId,
                name: studentName
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification(`Attendance marked for ${studentName}`, 'success');
            
            // Add to log
            addToLog({ student_id: studentId, name: studentName }, false);
            
            // Close modal and reset form
            closeManualModal();
            
            // Refresh log
            loadAttendanceLog();
            
        } else {
            showNotification(result.message || 'Failed to mark attendance', 'warning');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error marking attendance', 'error');
    }
}

// Attach form submit handler
document.addEventListener('DOMContentLoaded', function() {
    const manualForm = document.getElementById('manual-attendance-form');
    if (manualForm) {
        manualForm.addEventListener('submit', handleManualAttendance);
    }
});

function refreshAttendance() {
    if (currentPage === 'recognize') {
        loadAttendanceLog();
    } else if (currentPage === 'attendance') {
        location.reload();
    }
}

// ===================
// REGISTRATION PAGE
// ===================

function setupRegistrationPage() {
    // Image preview
    const imageInput = document.getElementById('face_image');
    if (imageInput) {
        imageInput.addEventListener('change', previewImage);
    }
    
    // Form validation
    const form = document.getElementById('registration-form');
    if (form) {
        form.addEventListener('submit', validateRegistrationForm);
    }
}

function previewImage(event) {
    const input = event.target;
    const preview = document.getElementById('image-preview');
    
    if (preview) {
        preview.innerHTML = '';
        preview.style.display = 'block';
        
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.style.maxWidth = '200px';
                img.style.maxHeight = '200px';
                img.style.borderRadius = '8px';
                img.style.marginTop = '10px';
                preview.appendChild(img);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }
}

function validateRegistrationForm(event) {
    const form = event.target;
    const studentId = form.querySelector('#student_id');
    const name = form.querySelector('#name');
    const department = form.querySelector('#department');
    const semester = form.querySelector('#semester');
    const faceImage = form.querySelector('#face_image');
    
    let isValid = true;
    
    // Clear previous errors
    clearFormErrors(form);
    
    // Validate Student ID
    if (!studentId.value.trim()) {
        showFormError(studentId, 'Student ID is required');
        isValid = false;
    }
    
    // Validate Name
    if (!name.value.trim()) {
        showFormError(name, 'Name is required');
        isValid = false;
    }
    
    // Validate Department
    if (!department.value) {
        showFormError(department, 'Please select a department');
        isValid = false;
    }
    
    // Validate Semester
    if (!semester.value) {
        showFormError(semester, 'Please select a semester');
        isValid = false;
    }
    
    // Validate Image
    if (!faceImage.files || faceImage.files.length === 0) {
        showFormError(faceImage, 'Please select a face photo');
        isValid = false;
    } else {
        const file = faceImage.files[0];
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        const maxSize = 5 * 1024 * 1024; // 5MB
        
        if (!validTypes.includes(file.type)) {
            showFormError(faceImage, 'Please upload a JPG or PNG image');
            isValid = false;
        }
        
        if (file.size > maxSize) {
            showFormError(faceImage, 'Image size should be less than 5MB');
            isValid = false;
        }
    }
    
    if (!isValid) {
        event.preventDefault();
        showNotification('Please fix the errors in the form', 'error');
    }
}

function showFormError(element, message) {
    const formGroup = element.closest('.form-group');
    if (formGroup) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'form-error';
        errorDiv.style.color = '#dc3545';
        errorDiv.style.fontSize = '0.85rem';
        errorDiv.style.marginTop = '5px';
        errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
        formGroup.appendChild(errorDiv);
        
        element.style.borderColor = '#dc3545';
    }
}

function clearFormErrors(form) {
    const errors = form.querySelectorAll('.form-error');
    errors.forEach(error => error.remove());
    
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.style.borderColor = '#e9ecef';
    });
}

// ===================
// ATTENDANCE PAGE
// ===================

function setupAttendancePage() {
    // Setup export button
    const exportBtn = document.querySelector('.btn-success');
    if (exportBtn && exportBtn.textContent.includes('Export CSV')) {
        exportBtn.addEventListener('click', exportTodayAttendance);
    }
    
    // Setup refresh button
    const refreshBtn = document.querySelector('.btn-secondary');
    if (refreshBtn && refreshBtn.textContent.includes('Refresh')) {
        refreshBtn.addEventListener('click', refreshAttendance);
    }
}

async function exportTodayAttendance() {
    try {
        showNotification('Preparing export...', 'info');
        
        const response = await fetch('/api/export/today', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            // Create blob and download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `attendance_${new Date().toISOString().split('T')[0]}.xlsx`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            showNotification('Export downloaded successfully', 'success');
        } else {
            const error = await response.json();
            showNotification(error.error || 'Export failed', 'error');
        }
    } catch (error) {
        console.error('Export error:', error);
        showNotification('Export failed. Please try again.', 'error');
    }
}

// ===================
// REPORT PAGE
// ===================

function setupReportPage() {
    // Set default dates
    const today = new Date().toISOString().split('T')[0];
    const firstDay = new Date();
    firstDay.setDate(1);
    const firstDayStr = firstDay.toISOString().split('T')[0];
    
    const startDate = document.getElementById('start_date');
    const endDate = document.getElementById('end_date');
    
    if (startDate && !startDate.value) {
        startDate.value = firstDayStr;
    }
    if (endDate && !endDate.value) {
        endDate.value = today;
    }
    
    // Setup form submission
    const form = document.getElementById('report-form');
    if (form) {
        form.addEventListener('submit', handleReportGeneration);
    }
    
    // Setup export button
    const exportBtn = document.querySelector('.btn-success');
    if (exportBtn && exportBtn.textContent.includes('Export to Excel')) {
        exportBtn.addEventListener('click', exportReport);
    }
    
    // Setup print button
    const printBtn = document.querySelector('.btn-secondary');
    if (printBtn && printBtn.textContent.includes('Print Report')) {
        printBtn.addEventListener('click', printReport);
    }
    
    // Initialize percentage bars
    setTimeout(initializePercentageBars, 100);
}

async function handleReportGeneration(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitBtn = document.getElementById('generate-report-btn');
    const loadingIndicator = document.getElementById('report-loading');
    
    if (submitBtn && loadingIndicator) {
        submitBtn.style.display = 'none';
        loadingIndicator.style.display = 'inline-block';
    }
    
    // Submit form normally - FastAPI will handle the POST request
}

function initializePercentageBars() {
    const percentageFills = document.querySelectorAll('.percentage-fill');
    percentageFills.forEach(fill => {
        const percentage = parseFloat(fill.dataset.percentage) || 0;
        
        // Set color class based on percentage
        if (percentage >= 75) {
            fill.classList.add('high');
        } else if (percentage >= 50) {
            fill.classList.add('medium');
        } else {
            fill.classList.add('low');
        }
        
        // Animate the fill
        fill.style.width = '0%';
        setTimeout(() => {
            fill.style.width = percentage + '%';
        }, 100);
    });
}

async function exportReport() {
    try {
        const startDate = document.getElementById('start_date').value;
        const endDate = document.getElementById('end_date').value;
        
        if (!startDate || !endDate) {
            showNotification('Please select date range first', 'warning');
            return;
        }
        
        showNotification('Generating report...', 'info');
        
        const response = await fetch('/api/export/attendance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                start_date: startDate,
                end_date: endDate
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `attendance_report_${startDate}_to_${endDate}.xlsx`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            showNotification('Report exported successfully', 'success');
        } else {
            const error = await response.json();
            showNotification(error.error || 'Export failed', 'error');
        }
    } catch (error) {
        console.error('Export error:', error);
        showNotification('Export failed. Please try again.', 'error');
    }
}

function printReport() {
    window.print();
}

// ===================
// STUDENTS PAGE
// ===================

function setupStudentsPage() {
    // Load students
    loadStudents();
    
    // Setup search functionality
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        searchInput.addEventListener('input', filterStudents);
    }
}

async function loadStudents() {
    try {
        const response = await fetch('/api/students');
        if (response.ok) {
            const students = await response.json();
            const tableBody = document.querySelector('.data-table tbody');
            
            if (tableBody) {
                tableBody.innerHTML = '';
                
                if (students.length === 0) {
                    tableBody.innerHTML = `
                        <tr>
                            <td colspan="7" class="text-center" style="padding: 40px; color: #6c757d;">
                                <i class="fas fa-users fa-2x mb-3"></i>
                                <p>No students registered yet</p>
                            </td>
                        </tr>
                    `;
                    return;
                }
                
                students.forEach(student => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${student.student_id}</td>
                        <td>${student.name}</td>
                        <td>${student.email || 'N/A'}</td>
                        <td>${student.department || 'N/A'}</td>
                        <td>${student.semester || 'N/A'}</td>
                        <td>${new Date(student.registration_date).toLocaleDateString()}</td>
                        <td class="actions">
                            <button class="btn-action view-btn" title="View Details" onclick="viewStudent('${student.student_id}')">
                                <i class="fas fa-eye"></i>
                            </button>
                            <button class="btn-action edit-btn" title="Edit" onclick="editStudent('${student.student_id}')">
                                <i class="fas fa-edit"></i>
                            </button>
                            <button class="btn-action delete-btn" title="Delete" onclick="deleteStudent('${student.student_id}', '${student.name}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </td>
                    `;
                    tableBody.appendChild(row);
                });
            }
        }
    } catch (error) {
        console.error('Failed to load students:', error);
    }
}

function filterStudents(event) {
    const searchTerm = event.target.value.toLowerCase();
    const rows = document.querySelectorAll('.data-table tbody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(searchTerm) ? '' : 'none';
    });
}

async function viewStudent(studentId) {
    try {
        const response = await fetch(`/api/students/${studentId}`);
        if (response.ok) {
            const student = await response.json();
            
            // Show modal with student details
            const modal = document.getElementById('student-details-modal');
            const content = document.getElementById('student-details-content');
            
            if (modal && content) {
                content.innerHTML = `
                    <div class="student-details">
                        <div class="detail-item">
                            <strong>Student ID:</strong> ${student.student_id}
                        </div>
                        <div class="detail-item">
                            <strong>Name:</strong> ${student.name}
                        </div>
                        <div class="detail-item">
                            <strong>Email:</strong> ${student.email || 'N/A'}
                        </div>
                        <div class="detail-item">
                            <strong>Department:</strong> ${student.department || 'N/A'}
                        </div>
                        <div class="detail-item">
                            <strong>Semester:</strong> ${student.semester || 'N/A'}
                        </div>
                        <div class="detail-item">
                            <strong>Registered:</strong> ${new Date(student.registration_date).toLocaleDateString()}
                        </div>
                        <div class="detail-actions mt-4">
                            <button class="btn btn-primary" onclick="editStudent('${student.student_id}')">
                                <i class="fas fa-edit"></i> Edit Student
                            </button>
                        </div>
                    </div>
                `;
                
                modal.style.display = 'flex';
            }
        }
    } catch (error) {
        console.error('Failed to load student:', error);
        showNotification('Failed to load student details', 'error');
    }
}

async function editStudent(studentId) {
    try {
        const response = await fetch(`/api/students/${studentId}`);
        if (response.ok) {
            const student = await response.json();
            
            // Show edit form in modal
            const modal = document.getElementById('student-details-modal');
            const content = document.getElementById('student-details-content');
            
            if (modal && content) {
                content.innerHTML = `
                    <form id="edit-student-form">
                        <div class="form-group">
                            <label>Student ID</label>
                            <input type="text" value="${student.student_id}" disabled class="form-control">
                        </div>
                        <div class="form-group">
                            <label>Name *</label>
                            <input type="text" id="edit-name" value="${student.name}" required class="form-control">
                        </div>
                        <div class="form-group">
                            <label>Email</label>
                            <input type="email" id="edit-email" value="${student.email || ''}" class="form-control">
                        </div>
                        <div class="form-group">
                            <label>Department *</label>
                            <select id="edit-department" required class="form-control">
                                <option value="">Select Department</option>
                                <option value="Computer Science" ${student.department === 'Computer Science' ? 'selected' : ''}>Computer Science</option>
                                <option value="Information Technology" ${student.department === 'Information Technology' ? 'selected' : ''}>Information Technology</option>
                                <option value="Electronics" ${student.department === 'Electronics' ? 'selected' : ''}>Electronics</option>
                                <option value="Mechanical" ${student.department === 'Mechanical' ? 'selected' : ''}>Mechanical</option>
                                <option value="Civil" ${student.department === 'Civil' ? 'selected' : ''}>Civil</option>
                                <option value="Electrical" ${student.department === 'Electrical' ? 'selected' : ''}>Electrical</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Semester *</label>
                            <select id="edit-semester" required class="form-control">
                                <option value="">Select Semester</option>
                                ${Array.from({length: 8}, (_, i) => 
                                    `<option value="Semester ${i+1}" ${student.semester === `Semester ${i+1}` ? 'selected' : ''}>
                                        Semester ${i+1}
                                    </option>`
                                ).join('')}
                            </select>
                        </div>
                        <div class="form-actions">
                            <button type="submit" class="btn btn-primary">
                                <i class="fas fa-save"></i> Save Changes
                            </button>
                            <button type="button" class="btn btn-secondary" onclick="closeStudentModal()">
                                <i class="fas fa-times"></i> Cancel
                            </button>
                        </div>
                    </form>
                `;
                
                // Add form submit handler
                setTimeout(() => {
                    const form = document.getElementById('edit-student-form');
                    if (form) {
                        form.addEventListener('submit', async (e) => {
                            e.preventDefault();
                            
                            const updatedStudent = {
                                name: document.getElementById('edit-name').value,
                                email: document.getElementById('edit-email').value,
                                department: document.getElementById('edit-department').value,
                                semester: document.getElementById('edit-semester').value
                            };
                            
                            try {
                                const updateResponse = await fetch(`/api/students/${studentId}`, {
                                    method: 'PUT',
                                    headers: {
                                        'Content-Type': 'application/json'
                                    },
                                    body: JSON.stringify(updatedStudent)
                                });
                                
                                const result = await updateResponse.json();
                                
                                if (result.success) {
                                    showNotification('Student updated successfully', 'success');
                                    closeStudentModal();
                                    loadStudents(); // Refresh the list
                                } else {
                                    showNotification(result.message || 'Update failed', 'error');
                                }
                            } catch (error) {
                                showNotification('Failed to update student', 'error');
                            }
                        });
                    }
                }, 100);
                
                modal.style.display = 'flex';
            }
        }
    } catch (error) {
        console.error('Failed to load student for edit:', error);
        showNotification('Failed to load student details', 'error');
    }
}

async function deleteStudent(studentId, studentName) {
    if (confirm(`Are you sure you want to delete student ${studentName} (${studentId})?\n\nThis will also delete all attendance records for this student.`)) {
        try {
            const response = await fetch(`/api/students/${studentId}`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            if (result.success) {
                showNotification('Student deleted successfully', 'success');
                loadStudents(); // Refresh the list
            } else {
                showNotification(result.message || 'Delete failed', 'error');
            }
        } catch (error) {
            console.error('Failed to delete student:', error);
            showNotification('Failed to delete student', 'error');
        }
    }
}

function closeStudentModal() {
    const modal = document.getElementById('student-details-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Close modal when clicking outside
window.addEventListener('click', function(event) {
    const modal = document.getElementById('manual-attendance-modal');
    if (modal && event.target === modal) {
        closeManualModal();
    }
    
    const studentModal = document.getElementById('student-details-modal');
    if (studentModal && event.target === studentModal) {
        closeStudentModal();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeManualModal();
        closeStudentModal();
    }
});

// ===================
// HELPER FUNCTIONS
// ===================

function showLoading(element) {
    if (element) {
        const originalContent = element.innerHTML;
        element.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        element.disabled = true;
        
        return () => {
            element.innerHTML = originalContent;
            element.disabled = false;
        };
    }
    return () => {};
}

// Initialize when page loads
window.addEventListener('load', function() {
    console.log("System fully loaded");
    
    // Add CSS for notifications if not already added
    if (!document.getElementById('notification-styles')) {
        const styles = document.createElement('style');
        styles.id = 'notification-styles';
        styles.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                border-radius: 8px;
                background: white;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                display: flex;
                align-items: center;
                gap: 12px;
                z-index: 9999;
                animation: slideInRight 0.3s ease;
                max-width: 400px;
                border-left: 4px solid #007bff;
            }
            
            .notification-success {
                border-left-color: #28a745;
            }
            
            .notification-error {
                border-left-color: #dc3545;
            }
            
            .notification-warning {
                border-left-color: #ffc107;
            }
            
            .notification-info {
                border-left-color: #17a2b8;
            }
            
            .notification button {
                background: none;
                border: none;
                font-size: 1.2rem;
                cursor: pointer;
                margin-left: auto;
                color: #666;
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            .form-error {
                color: #dc3545;
                font-size: 0.85rem;
                margin-top: 5px;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            @keyframes successPulse {
                0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }
                70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
                100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
            }
        `;
        document.head.appendChild(styles);
    }
});