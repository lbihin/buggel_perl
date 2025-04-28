document.addEventListener("DOMContentLoaded", function () {
    const shapeTypeSelect = document.getElementById("shapeType");
    const shapeControls = document.getElementById("shapeControls");

    if (shapeTypeSelect && shapeControls) {
        shapeTypeSelect.addEventListener("change", function () {
            const shapeType = shapeTypeSelect.value;
            fetch(`/shapes/${shapeType}/controls/`)
                .then((response) => response.text())
                .then((html) => {
                    shapeControls.innerHTML = html;
                })
                .catch((error) => console.error("Erreur:", error));
        });
    }

    // Écouteur d'événements pour le déclencheur HTMX lors de la mise à jour des dimensions
    document.body.addEventListener('shapeDimensionsUpdated', function (event) {
        // Récupérer l'ID de la forme mise à jour depuis l'événement si disponible
        const shapeId = event.detail ? event.detail.shapeId : null;

        // Rafraîchir le visualisateur - on peut cibler uniquement la forme modifiée si l'ID est disponible
        if (shapeId) {
            const shapePreview = document.querySelector(`.shape-preview[data-shape-id="${shapeId}"]`);
            if (shapePreview) {
                // Demander une mise à jour du visualisateur pour cette forme spécifique
                updateShapeVisualization(shapeId);
            }
        } else {
            // Si pas d'ID spécifique, rafraîchir tous les visualisateurs
            updateAllShapeVisualizations();
        }
    });

    // Écouteur pour le changement de type de forme
    document.body.addEventListener('shapeTypeChanged', function (event) {
        // Similaire à shapeDimensionsUpdated mais pourrait avoir des comportements spécifiques
        // Pour l'instant, même comportement
        const shapeId = event.detail ? event.detail.shapeId : null;
        if (shapeId) {
            updateShapeVisualization(shapeId);
        } else {
            updateAllShapeVisualizations();
        }
    });
});

/**
 * Rafraîchit le visualisateur pour une forme spécifique
 * @param {string|number} shapeId - ID de la forme à mettre à jour
 */
function updateShapeVisualization(shapeId) {
    const shapeCard = document.querySelector(`.card[data-shape-id="${shapeId}"]`);
    if (shapeCard) {
        const previewContainer = shapeCard.querySelector('.shape-preview');
        if (previewContainer) {
            // Ajouter une classe pour indiquer que le chargement est en cours
            previewContainer.classList.add('refreshing');

            // Appeler l'API pour obtenir les nouvelles données de visualisation
            fetch(`/shapes/${shapeId}/preview/`)
                .then(response => response.text())
                .then(html => {
                    previewContainer.innerHTML = html;
                    previewContainer.classList.remove('refreshing');
                })
                .catch(error => {
                    console.error('Erreur lors de la mise à jour du visualisateur:', error);
                    previewContainer.classList.remove('refreshing');
                });
        }
    }
}

/**
 * Rafraîchit tous les visualisateurs de formes sur la page
 */
function updateAllShapeVisualizations() {
    document.querySelectorAll('.shape-preview').forEach(preview => {
        const shapeId = preview.closest('.card').dataset.shapeId;
        if (shapeId) {
            updateShapeVisualization(shapeId);
        }
    });
}
