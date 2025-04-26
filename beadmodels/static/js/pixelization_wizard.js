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

    // Fonction pour mettre à jour les dimensions selon l'élément sélectionné
    function updateDimensions(element) {
        const width = parseInt(element.getAttribute('data-width'));
        const height = parseInt(element.getAttribute('data-height'));

        if (width && height) {
            document.getElementById('id_grid_width').value = width;
            document.getElementById('id_grid_height').value = height;

            // Check if these elements exist before trying to update them
            const displayWidth = document.getElementById('display_width');
            const displayHeight = document.getElementById('display_height');

            if (displayWidth) {
                displayWidth.textContent = width;
            }

            if (displayHeight) {
                displayHeight.textContent = height;
            }
        }
    }

    // Gestion des clics sur les formes personnalisées
    const shapeSelectors = document.querySelectorAll('.shape-selector');
    shapeSelectors.forEach(selector => {
        selector.addEventListener('change', function () {
            if (this.checked) {
                updateDimensions(this);
            }
        });

        // Initialisation si une forme est déjà sélectionnée
        if (selector.checked) {
            updateDimensions(selector);
        }
    });

    // Gestion des clics sur les grilles prédéfinies (obsolète mais maintenu pour compatibilité)
    const boardSelectors = document.querySelectorAll('input[name="board_id"]');
    boardSelectors.forEach(selector => {
        selector.addEventListener('change', function () {
            if (this.checked) {
                updateDimensions(this);
            }
        });

        // Initialisation si une grille est déjà sélectionnée
        if (selector.checked) {
            updateDimensions(selector);
        }
    });

    // Validation du formulaire
    const form = document.querySelector('.needs-validation');
    if (form) {
        form.addEventListener('submit', function (event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();

                // Ajouter un message d'erreur si aucun n'existe
                if (document.querySelector('.alert-danger') === null) {
                    const errorAlert = document.createElement('div');
                    errorAlert.className = 'alert alert-danger mt-3';
                    errorAlert.innerHTML = '<i class="bi bi-exclamation-triangle"></i> Veuillez vérifier les champs du formulaire et réessayer.';
                    form.prepend(errorAlert);
                }
            } else {
                // Le formulaire est valide, afficher l'overlay de chargement
                showLoading("Création du modèle de perles en cours...");
            }

            form.classList.add('was-validated');
        });
    }
});
