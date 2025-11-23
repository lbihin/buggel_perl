// Small, focused helper for the shapes create/edit form.
// Purpose: keep JS minimal and well-documented. Only handles UI niceties
// (show/hide dimension blocks and preview canvas updates). Core behavior
// and data handling remain server-driven via HTMX.

document.addEventListener('DOMContentLoaded', function () {
  // Find selectors and fields using conservative queries so this script
  // doesn't throw in contexts where fields are not present.
  const form = document.querySelector('#shapeForm');
  if (!form) return;

  const dimensionSelectors = form.querySelectorAll('.dimension-selector');
  const dimensionFields = form.querySelectorAll('.dimension-fields');
  const autoTypeLabel = form.querySelector('#auto_type_label');
  const shapeTypeHidden = form.querySelector('#id_shape_type');
  const previewCanvas = form.querySelector('#previewCanvas');

  // Toggle fields based on selected radio. Keep behavior local to the form.
  function showFieldsFor(type) {
    dimensionFields.forEach(field => field.classList.add('d-none'));
    const target = form.querySelector(`.${type}-fields`);
    if (target) target.classList.remove('d-none');
  }

  function updateAutoTypeLabel(type) {
    if (!autoTypeLabel) return;
    let text = 'Sera déterminé automatiquement';
    if (type === 'rectangle') text = 'Rectangle';
    else if (type === 'square') text = 'Carré';
    else if (type === 'circle') text = 'Cercle';
    autoTypeLabel.textContent = text;
  }

  // Attach listeners to radio buttons to update UI only (no data-saving).
  dimensionSelectors.forEach(selector => {
    selector.addEventListener('change', function () {
      const type = this.value;
      showFieldsFor(type);
      if (shapeTypeHidden) shapeTypeHidden.value = type;
      updateAutoTypeLabel(type);

      // If there's a preview canvas, trigger an update (user may provide
      // implementation by adding a global `updateShapePreview` function
      // or extending this file). Keep the call safe.
      if (typeof window.updateShapePreview === 'function') {
        window.updateShapePreview();
      }
    });
  });

  // Wire inputs to preview if a preview function exists.
  if (previewCanvas && typeof window.updateShapePreview === 'function') {
    ['id_width', 'id_height', 'id_size', 'id_diameter'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', window.updateShapePreview);
    });

    // Initial preview call (safe-guarded)
    if (typeof window.updateShapePreview === 'function') window.updateShapePreview();
  }
});
