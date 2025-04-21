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

// Fonction pour supprimer une forme
function deleteShape(shapeId) {
    if (!confirm('Êtes-vous sûr de vouloir supprimer cette forme ?')) {
        return;
    }

    // Créer un formulaire pour envoyer la requête POST
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = `/settings/delete-shape/${shapeId}/`;

    // Ajouter le token CSRF
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
    const csrfInput = document.createElement('input');
    csrfInput.type = 'hidden';
    csrfInput.name = 'csrfmiddlewaretoken';
    csrfInput.value = csrfToken;

    form.appendChild(csrfInput);
    document.body.appendChild(form);

    // Soumettre le formulaire
    form.submit();
}
