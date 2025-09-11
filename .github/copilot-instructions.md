# Buggel - AI Assistant Instructions

## Project Overview
Buggel is a Django web application for creating iron-on bead patterns from images. The application transforms images into pixelated grids suitable for crafting with iron-on beads.

## Architecture

### Core Components
1. **Models and Relationships**:
   - `BeadModel`: Main entity representing a bead pattern project
   - `Bead`: Represents individual bead colors with RGB values and quantity
   - `BeadBoard`: Defines board templates with dimensions for patterns
   - `BeadShape` (in shapes app): Custom shapes for bead arrangements

2. **Wizard Framework**:
   - Custom modular wizard system for multi-step processes
   - Base classes: `BaseWizard` and `WizardStep` in `beadmodels/wizards.py`
   - Implementation: `PixelizationWizard` in `beadmodels/pixelization_wizard.py`
   - Session-based state management for wizard progression

3. **Image Processing**:
   - Pixelization algorithm using OpenCV and scikit-learn
   - Color reduction through k-means clustering
   - Custom palette matching for available beads

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

## Key Patterns & Conventions

### Frontend Structure
- Templates inherit from `base.html`
- Static assets organized by component in `beadmodels/static/`
- HTMX used for dynamic UI updates
- Bootstrap 5 with crispy forms for form rendering

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
Users can define their own bead colors which are then used for color matching in the pixelization process.

### Shape System
The shapes app defines custom grid layouts that integrate with the pixelization wizard.

### Media Files
Original images stored in `media/originals/` with processed patterns in `media/patterns/`.