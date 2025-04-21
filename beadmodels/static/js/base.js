document.addEventListener("DOMContentLoaded", function () {
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll(".alert-dismissible");
    alerts.forEach((alert) => {
        setTimeout(() => {
            alert.classList.add("fade");
            setTimeout(() => alert.remove(), 500);
        }, 5000);
    });
});
