document.addEventListener('DOMContentLoaded', function () {
    // Fermeture du formulaire d'ajout/édition
    document.body.addEventListener('click', function (e) {
    if (e.target.classList.contains('cancel-form-btn')) {
        document.getElementById('shape-form-container').innerHTML = ''
    }
    })

    // Animation d'entrée/sortie pour les éléments HTMX
    document.body.addEventListener('htmx:afterSwap', function (event) {
    if (event.detail.target.id === 'shape-form-container') {
        event.detail.target.querySelectorAll('.shape-form').forEach((form) => {
        form.classList.add('animate__animated', 'animate__fadeIn')
        })
    }
    })
})

// Fonction pour afficher le formulaire d'ajout de forme directement dans le tableau
function showAddShapeForm() {
    // Vérifie si le formulaire est déjà présent pour éviter les doublons
    if (document.getElementById('inline-add-shape-form')) {
    return
    }

    // Si le message "Aucune forme disponible" est affiché, on le cache
    const emptyStateRow = document.getElementById('empty-state-row')
    if (emptyStateRow) {
    emptyStateRow.style.display = 'none'
    }

    // On insère le formulaire avant le placeholder
    const placeholder = document.getElementById('inline-form-placeholder') || document.querySelector('.shapes-table tbody')

    // On va ajouter le formulaire en utilisant HTMX
    htmx.ajax('GET', "{% url 'shapes:get_inline_add_form' %}", {
    target: placeholder,
    swap: 'beforebegin'
    })

    // Faire défiler jusqu'à la nouvelle ligne avec une petite animation
    setTimeout(() => {
    const newRow = document.getElementById('inline-add-shape-form')
    if (newRow) {
        newRow.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
        })
        // Focalise le premier champ pour faciliter la saisie
        newRow.querySelector('input[name="name"]').focus()
    }
    }, 100)
}

// Fonction pour annuler l'ajout en ligne
function cancelInlineAdd() {
    const row = document.getElementById('inline-add-shape-form')
    if (row) {
    row.remove()

    // Vérifier s'il n'y a pas de formes pour réafficher le message vide
    const shapes = document.querySelectorAll('.shape-row:not(.new-shape-row)')
    if (shapes.length === 0) {
        const emptyStateRow = document.getElementById('empty-state-row')
        if (emptyStateRow) {
        emptyStateRow.style.display = ''
        }
    }
    }
}

// Écouter les événements HTMX pour gérer l'affichage après une action
document.body.addEventListener('htmx:afterSwap', function (event) {
    // Si on vient de charger la liste et qu'il n'y a plus de formulaire d'ajout,
    // on réaffiche l'état vide si nécessaire
    if (event.detail.target.id === 'shape-list-container') {
    if (!document.getElementById('inline-add-shape-form')) {
        const shapes = document.querySelectorAll('.shape-row:not(.new-shape-row)')
        const emptyStateRow = document.getElementById('empty-state-row')

        if (shapes.length === 0 && emptyStateRow) {
        emptyStateRow.style.display = ''
        }
    }
    }
})