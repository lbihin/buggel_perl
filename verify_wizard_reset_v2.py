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

print("Test 1: Direct external navigation should reset...")
# Simulate user mid-wizard at step 2
session = client.session
session["model_creation_wizard_step"] = 2
session.save()

# Perform direct GET from external referer (should trigger auto-reset)
resp = client.get(
    "/beadmodels/model-creation/",
    HTTP_REFERER="http://testserver/beadmodels/my-models/",
)
assert resp.status_code in (302, 200), f"Unexpected status code: {resp.status_code}"

# Follow redirect if present
if resp.status_code == 302:
    location = resp.url if hasattr(resp, "url") else resp["Location"]
    resp = client.get(location)

content = resp.content.decode("utf-8")
assert (
    "Étape 1" in content or "Etape 1" in content
), "Step 1 marker not found after external navigation"

# Confirm session step reset
session = client.session
step = session.get("model_creation_wizard_step")
assert step == 1, f"Wizard step not reset after external navigation, found {step}"
print("✓ Test 1 PASSED: External navigation resets to step 1")

print("\nTest 2: Internal wizard navigation should NOT reset...")
# Set up for step 2
session = client.session
session["model_creation_wizard_step"] = 2
session["model_creation_wizard"] = {"image_data": {"image_path": "test.png"}}
session.save()

# Navigate from wizard itself (internal referer - should NOT reset)
resp = client.get(
    "/beadmodels/model-creation/",
    HTTP_REFERER="http://testserver/beadmodels/model-creation/",
)
assert resp.status_code in (302, 200), f"Unexpected status code: {resp.status_code}"

if resp.status_code == 302:
    location = resp.url if hasattr(resp, "url") else resp["Location"]
    resp = client.get(
        location, HTTP_REFERER="http://testserver/beadmodels/model-creation/"
    )

# Should still be at step 2
session = client.session
step = session.get("model_creation_wizard_step")
assert (
    step == 2
), f"Wizard step incorrectly reset during internal navigation, found {step}"
print("✓ Test 2 PASSED: Internal navigation preserves wizard state")

print("\n✅ All wizard auto-reset verification tests PASSED")
