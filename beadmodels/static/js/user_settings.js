document.addEventListener('DOMContentLoaded', function () {
    // Auto-dismiss success messages after 5 seconds
    const successAlerts = document.querySelectorAll('.alert-success[data-auto-dismiss="success"]');
    successAlerts.forEach(function (alert) {
        setTimeout(function () {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
