from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import Client

# Minimal verification script for wizard auto-reset behavior

# Allow test client host
if "testserver" not in settings.ALLOWED_HOSTS:
    settings.ALLOWED_HOSTS.append("testserver")

User = get_user_model()
username = "wizard_tester"
password = "pass1234"
email = "wizard_tester@example.com"

if not User.objects.filter(username=username).exists():
    User.objects.create_user(username=username, email=email, password=password)

client = Client()
logged_in = client.login(username=username, password=password)
assert logged_in, "Login failed"

# Simulate user mid-wizard at step 2
session = client.session
session["model_creation_wizard_step"] = 2
session.save()

# Perform direct GET to wizard URL (should trigger auto-reset)
resp = client.get("/beadmodels/model-creation/")
assert resp.status_code in (302, 200), f"Unexpected status code: {resp.status_code}"

# Follow redirect if present
if resp.status_code == 302 and "Location" in resp.headers:
    resp = client.get(resp.headers["Location"])

content = resp.content.decode("utf-8")
# Check that we are on step 1 (progress bar and heading)
assert (
    "Étape 1" in content or "Etape 1" in content
), "Step 1 marker not found in response HTML"

# Confirm session step reset
session = client.session
step = session.get("model_creation_wizard_step")
assert step == 1, f"Wizard step not reset, found {step}"

print(
    "Wizard auto-reset verification PASSED: direct GET returned Step 1 with fresh state."
)
