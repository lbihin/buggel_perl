/**
 * Fichier d'utilitaires JavaScript communs pour l'application
 */

/**
 * Affiche l'overlay de chargement avec un message personnalisable
 * @param {string} message - Message à afficher pendant le chargement
 */
function showLoading(message = "Traitement en cours...") {
    const overlay = document.getElementById('loadingOverlay');
    if (!overlay) return;

    const textElement = overlay.querySelector('.loading-text');
    if (textElement) {
        textElement.textContent = message;
    }

    overlay.style.visibility = 'visible';
    overlay.style.opacity = '1';
    overlay.classList.add('show');
    document.body.style.overflow = 'hidden'; // Empêche le défilement pendant le chargement
}

/**
 * Masque l'overlay de chargement
 */
function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (!overlay) return;

    overlay.classList.remove('show');
    overlay.style.visibility = 'hidden';
    overlay.style.opacity = '0';
    document.body.style.overflow = ''; // Restaure le défilement
}