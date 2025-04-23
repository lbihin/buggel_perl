document.addEventListener('DOMContentLoaded', function () {
    // Fonction pour mettre à jour les dimensions selon l'élément sélectionné
    function updateDimensions(element) {
        const width = parseInt(element.getAttribute('data-width'));
        const height = parseInt(element.getAttribute('data-height'));

        if (width && height) {
            document.getElementById('id_grid_width').value = width;
            document.getElementById('id_grid_height').value = height;
            document.getElementById('display_width').textContent = width;
            document.getElementById('display_height').textContent = height;
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
            }

            form.classList.add('was-validated');
        });
    }
});
