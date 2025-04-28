document.addEventListener('DOMContentLoaded', function () {
    // Les fonctions showLoading() et hideLoading() sont maintenant disponibles globalement depuis utils.js

    // Gestion du téléchargement de l'image
    document.getElementById('downloadBtn')?.addEventListener('click', function () {
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
