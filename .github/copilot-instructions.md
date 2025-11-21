# Buggel - AI Assistant Instructions

## Project Overview
Buggel is a Django web application for creating iron-on bead patterns from images. The application transforms images into pixelated grids suitable for crafting with iron-on beads.

## Architecture

### Core Components
1. **Models and Relationships**:
   - `BeadModel`: Main entity representing a bead pattern project with original image and processed pattern
   - `Bead`: Represents individual bead colors with RGB values and quantity tracking
   - `BeadBoard`: Defines board templates with dimensions for patterns
   - `BeadShape` (in shapes app): Custom shapes for bead arrangements (rectangle, square, circle)

2. **Wizard Framework**:
   - Custom modular wizard system for multi-step processes
   - Base classes: `BaseWizard` and `WizardStep` in `beadmodels/wizards.py`
   - Implementation: `PixelizationWizard` in `beadmodels/pixelization_wizard.py`
   - Session-based state management for wizard progression
   - Two main wizards: `ModelCreationWizard` and `PixelizationWizard`

3. **Image Processing**:
   - Pixelization algorithm using OpenCV and scikit-learn
   - Color reduction through k-means clustering
   - Custom palette matching for available beads
   - Implementation in `beadmodels/pixelization_wizard.py`
   - Processing steps: resize → reduce colors → match palette → generate grid

## Development Workflows

### Environment Setup
```bash
# Install dependencies with Poetry
poetry install

# Run migrations
poetry run python manage.py migrate

# Create superuser
poetry run python manage.py createsuperuser

# Start development server
poetry run python manage.py runserver
```

### Testing
```bash
# Run all tests in parallel
poetry run pytest

# Run specific test file
poetry run pytest beadmodels/tests.py

# Run with specific marker
poetry run pytest -m "not slow"
```

### Session Management
```bash
# Reset all wizard sessions
poetry run python reset_wizard.py

# Check beads in the database
poetry run python check_beads.py
```

## Key Patterns & Conventions

### Frontend Structure
- Templates inherit from `base.html`
- Static assets organized by component in `beadmodels/static/`
- HTMX used for dynamic UI updates
- Bootstrap 5 with crispy forms for form rendering
- French is the primary language for UI elements and code comments

### Wizard Pattern
When implementing new wizards:
1. Extend `LoginRequiredWizard` from `beadmodels/wizards.py`
2. Create step classes extending `WizardStep`
3. Define `handle_get` and `handle_post` methods for each step
4. Use `wizard.get_data()` and `wizard.update_data()` for state management

Example from `pixelization_wizard.py`:
```python
class PixelizationWizard(LoginRequiredWizard):
    name = "Assistant de Pixelisation"
    steps = [ConfigurationStep, ResultStep, SaveStep]
    session_key = "pixelization_wizard"

    def get_url_name(self):
        return "beadmodels:pixelization_wizard"

    def finish_wizard(self):
        # Logic for wizard completion
        self.reset_wizard()
        return redirect("beadmodels:home")
```

### Image Processing
Images are processed in stages:
1. Image upload/selection
2. Resizing to match bead board dimensions
3. Color reduction using k-means clustering
4. Optional color matching with user's available bead colors
5. Grid generation for bead placement visualization

## Critical Integration Points

### Custom User Beads
Users can define their own bead colors which are then used for color matching in the pixelization process:
- `Bead` model stores RGB values and quantities
- `BeadForm` for adding/editing beads
- Color matching algorithm prioritizes user's available beads
- Bead management views are in `beadmodels/views.py` (BeadListView, BeadCreateView, etc.)

### Shape System
The shapes app defines custom grid layouts that integrate with the pixelization wizard:
- `BeadShape` model defines geometric properties (rectangle, square, circle)
- Shapes can be user-created or system defaults
- Shape selection affects grid dimensions and pattern output
- Integration happens during the configuration step of the pixelization wizard

### HTMX Integration
HTMX is used for dynamic UI updates without full page reloads:
- Middleware: `django_htmx.middleware.HtmxMiddleware`
- HTMX attributes in forms for live updates (see form widgets)
- Dedicated HTMX view functions with `_htmx` suffix (e.g., `bead_edit_quantity_htmx`)
- Response partials return only the necessary HTML fragments

### Media Files
- Original images stored in `media/originals/`
- Processed patterns in `media/patterns/`
- Static assets in `beadmodels/static/` organized by component (CSS, JS, images)
- Media URLs configured only in development mode

### AppPreferencesMiddleware
- Middleware injects user preferences into the request
- Defined in `beadmodels/middleware.py`
- Allows for consistent user settings across the application
- Used by the wizards for default values and behavior

### Experimentation
- Jupyter notebooks in the `notebooks/` directory contain experimental algorithms
- `test_pixelisation.ipynb` is the main notebook for image processing experiments
- Use these to test new processing techniques before implementation
- Experimental code should be refactored into appropriate modules when ready