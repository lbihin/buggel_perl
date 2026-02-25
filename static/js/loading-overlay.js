/**
 * Loading overlay for HTMX requests.
 * Shows/hides a #loadingOverlay element during HTMX requests.
 */
document.addEventListener('DOMContentLoaded', function () {
  const overlay = document.getElementById('loadingOverlay')
  if (!overlay) return

  document.body.addEventListener('htmx:beforeSend', function () {
    overlay.classList.remove('d-none')
  })

  document.body.addEventListener('htmx:afterOnLoad', function () {
    overlay.classList.add('d-none')
  })
})
