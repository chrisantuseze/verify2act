# GCP / Vertex AI Setup Guide for Verify2Act

This guide sets up Application Default Credentials (ADC) so the VLM planner
can call `gemini-2.5-flash` via the **Agent Platform (Vertex AI) API** using
your GCP project credits — bypassing the rate limits of a free AI Studio key.

> **Project ID**: `verify2act`  
> **GCP account**: console.cloud.google.com (billing account: My Billing Account)

---

## Prerequisites (one-time, already done)

These steps only need to be done once in the GCP Console — they are already
set up for the `verify2act` project.

### Billing Credits
Navigate to **Billing → Credits** to confirm credits are available:

![GCP Billing Credits](docs/gcp_setup_screenshots/01_billing_credits.png)

- ✅ **Trial credit for GenAI App Builder** — $1,000.00 available
- ✅ **Free Trial** — $299.75 available (expires September 2026)

### Agent Platform API
Navigate to `console.cloud.google.com/apis/library/aiplatform.googleapis.com?project=verify2act`
and confirm the API is enabled:

![Agent Platform API Enabled](docs/gcp_setup_screenshots/04_agent_platform_api_enabled.png)

> **Note**: GCP recently rebranded **Vertex AI → Agent Platform**. The underlying
> service name (`aiplatform.googleapis.com`) and API endpoint are unchanged.

### Why No Service Account JSON Key?
An org policy on this GCP organization **blocks service account key creation**:

![Service Account Key Disabled](docs/gcp_setup_screenshots/05_service_account_key_disabled.png)

This means you **cannot** download a JSON key file. You must authenticate as
yourself using `gcloud auth application-default login` instead.

### Why "Agent Platform User" Instead of "Vertex AI User"?
When searching for roles in the GCP Console, you will not find "Vertex AI User".
Search for `aiplatform.user` and select **Agent Platform User** — this is the
same role, just renamed:

![IAM Agent Platform User Role](docs/gcp_setup_screenshots/03_iam_agent_platform_user.png)

---

## Per-Machine Setup

Run these steps on **every new machine/server**.

### Step 1 — Install `google-auth` in your conda environment

```bash
conda activate verify2act
pip install google-auth
```

### Step 2 — Install the gcloud CLI

```bash
curl https://sdk.cloud.google.com | bash -s -- --disable-prompts
exec -l $SHELL   # reload shell to pick up the new PATH
```

Add to `~/.bashrc` so it persists across sessions:

```bash
echo 'source ~/google-cloud-sdk/path.bash.inc' >> ~/.bashrc
echo 'source ~/google-cloud-sdk/completion.bash.inc' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
gcloud version
```

### Step 3 — Authenticate with Application Default Credentials

Because this is a headless server, use the `--no-launch-browser` flag:

```bash
gcloud auth application-default login --no-launch-browser
```

1. Copy the long URL it prints and open it in **your local browser**
2. Log in with the Google account that owns the `verify2act` project
3. Copy the authorization code shown in the browser and paste it back in the terminal

You should see:
```
Credentials saved to file: [/home/<you>/.config/gcloud/application_default_credentials.json]
```

### Step 4 — Set the quota project

```bash
gcloud auth application-default set-quota-project verify2act
```

### Step 5 — Verify

```bash
python3 -c "
import google.auth
creds, proj = google.auth.default()
print('Credentials type:', type(creds).__name__)
print('quota_project_id:', getattr(creds, 'quota_project_id', 'NOT SET'))
"
```

Expected output:
```
Credentials type: Credentials
quota_project_id: verify2act
```

> `Project: None` is normal for user ADC credentials — the planner code falls
> back to `quota_project_id` automatically.

---

## Running Inference

No extra flags needed — the planner auto-detects ADC and uses Vertex AI:

```bash
conda activate verify2act

python3 verify2act/pipeline/inference_calvin.py \
  --train-folder calvin/models/hulc_baseline \
  --dataset-path calvin/dataset/task_ABCD_D_filtered \
  --low-level-policy diffusion \
  --low-level-policy-ckpt calvin/models/diffusion_baseline \
  --device cuda \
  --wm-mode vlm_only \
  --num-sequences 10
```

---

## How It Works

The planner (`verify2act/pipeline/planner.py`) checks in order:

1. `GEMINI_API_KEY` env var → uses AI Studio (free tier, **rate-limited**)
2. `~/.config/gcloud/application_default_credentials.json` → uses **Vertex AI** ✅
3. Neither found → raises `ValueError: GEMINI_API_KEY environment variable is not set`

The Vertex AI backend (`verify2act/pipeline/gemini_backend.py`) calls:
```
https://us-central1-aiplatform.googleapis.com/v1/projects/verify2act/
  locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent
```

Costs are billed against the `verify2act` GCP project credits.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Command 'gcloud' not found` | `source ~/google-cloud-sdk/path.bash.inc` or re-run Step 2 |
| `ModuleNotFoundError: google.auth` | `pip install google-auth` in the conda env (Step 1) |
| `ValueError: GEMINI_API_KEY ... not set` | ADC file missing — re-run Steps 3 & 4 |
| `DefaultCredentialsError` | Re-run Steps 3 & 4 |
| `quota_project_id: NOT SET` | Re-run Step 4 |
| `HTTP 403` on Vertex AI call | Confirm Agent Platform API is enabled (see Prerequisites) |
| `Project: None` in python check | Normal — code falls back to `quota_project_id` automatically |
| Can't find "Vertex AI User" role | Search `aiplatform.user` → select **Agent Platform User** instead |
| Service account key creation blocked | Expected — use `gcloud auth application-default login` (Step 3) |
