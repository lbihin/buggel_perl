document.addEventListener('DOMContentLoaded', function () {
    // Fonction pour afficher l'overlay de chargement
    function showLoading(message = "Traitement en cours...") {
        const overlay = document.getElementById('loadingOverlay');
        const textElement = overlay.querySelector('.loading-text');

        if (textElement) {
            textElement.textContent = message;
        }

        overlay.classList.add('show');
        document.body.style.overflow = 'hidden'; // Empêche le défilement pendant le chargement
    }

    // Fonction pour masquer l'overlay de chargement
    function hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.classList.remove('show');
        document.body.style.overflow = ''; // Restaure le défilement
    }

    // Gestion du téléchargement de l'image
    document.getElementById('downloadBtn').addEventListener('click', function () {
        const imageElement = document.getElementById('pixelizedImage');
        const imageURL = imageElement.getAttribute('src');

        const downloadLink = document.createElement('a');
        downloadLink.href = imageURL;
        downloadLink.download = 'modele_perles.png';
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    });

    // Gestion du bouton Précédent
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function () {
            showLoading("Retour aux paramètres du modèle...");
        });
    }
});
