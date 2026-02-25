/**
 * HTMX CSRF Configuration
 * Automatically injects the Django CSRF token into every HTMX request.
 */
document.body.addEventListener('htmx:configRequest', (event) => {
  const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')
  if (csrfToken) {
    event.detail.headers['X-CSRFToken'] = csrfToken.value
  }
})
