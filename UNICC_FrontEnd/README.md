# UNICC AI Safety Lab — Frontend (UNICC_FrontEnd)

**UI/Integration Lead:** Dennis Wu  
**Live demo:** https://unicc-ai-guard.replit.app  
**Backend it connects to:** `app/slm/api.py` (FastAPI, port 8000)

---

## Source code

The main application pages and components are maintained in the Replit project alongside this repository:

**Live app:** https://unicc-ai-guard.replit.app

The compiled shared libraries in `lib/` (`api-client-react`, `api-zod`, `db`) are the importable packages the Replit app depends on and are fully present in this repository. The Replit project imports them as workspace dependencies.

---

## What This Module Does

The UNICC Frontend is a React-based interface hosted on Replit that lets non-technical UNICC stakeholders evaluate any AI agent without touching the command line:

1. **Submit a GitHub URL** — paste any repo link and click Evaluate
2. **View three expert panels** — Governance, Threat, and Behavioral results displayed in distinct color-coded cards, each showing status, risk level, recommended action, rationale, and framework citations
3. **Read the final council verdict** — APPROVE / REVISE / ESCALATE / REJECT rendered in plain language with mitigation requirements
4. **Download the HTML report** — one-click formatted report suitable for UNICC stakeholder presentation

The frontend is a display and submission layer only. All evaluation logic runs in the FastAPI backend.

---

## Repository Structure

```
UNICC_FrontEnd/
├── lib/
│   ├── db/                    # Drizzle ORM schema — evaluation history persistence (SQLite)
│   ├── api-zod/               # Zod-validated API type definitions (generated from OpenAPI spec via orval)
│   └── api-client-react/      # React Query hooks for /health, /evaluate, /report, /smoke-test
├── scripts/
│   ├── post-merge.sh          # Git hook: rebuilds lib packages on branch merge
│   └── src/hello.ts           # Workspace smoke test
├── package.json               # pnpm workspace root
├── pnpm-workspace.yaml        # Declares lib/* as workspace packages
├── tsconfig.base.json         # Shared TypeScript compiler options
└── tsconfig.json              # App-level TypeScript config
```

The main application pages and components live in the Replit deployment and are bundled there. The files in this directory are the **shared library packages** the Replit app imports as workspace dependencies:

| Package | Purpose |
|---|---|
| `lib/api-client-react` | React Query hooks — `useHealthCheck`, `useEvaluate`, `useReport` |
| `lib/api-zod` | Zod schemas for runtime validation of all API responses |
| `lib/db` | Drizzle schema for persisting evaluation history (optional, SQLite) |

---

## Connecting the Replit Frontend to Your Local Backend

The Replit frontend cannot reach `localhost` directly. Use ngrok to expose the local backend over a public HTTPS URL.

### Step 1 — Start the backend locally

```bash
# From the repo root:
uvicorn app.slm.api:app --host 0.0.0.0 --port 8000
```

Verify it is running:
```bash
curl http://localhost:8000/health
```

### Step 2 — Expose via ngrok

```bash
ngrok http 8000
# Generates a URL like: https://xxxx-xx-xx-xx.ngrok-free.app
```

Copy the `https://` URL (not http).

### Step 3 — Set the API base URL in Replit

In the Replit project, open **Secrets** (padlock icon in the left sidebar) and set:

```
Key:   VITE_API_BASE_URL
Value: https://xxxx-xx-xx-xx.ngrok-free.app
```

### Step 4 — Refresh the frontend

Restart or hard-refresh the Replit app. All API calls now route to your local backend through the tunnel.

---

## Rebuilding the Library Packages Locally

Only needed if you modify `lib/api-zod` or `lib/api-client-react`:

```bash
cd UNICC_FrontEnd
pnpm install
pnpm build
```

---

## API Endpoints Used by the Frontend

| Endpoint | Method | Used for |
|---|---|---|
| `/health` | GET | Header status badge — shows backend connectivity |
| `/evaluate` | POST | Submit GitHub URL → full JSON council result |
| `/report` | POST | Submit GitHub URL → rendered HTML report (opens in new tab) |
| `/smoke-test` | GET | Dev tool — confirms all 3 modules are live |

---

## Submitting VeriMedia via the Frontend

1. Open https://unicc-ai-guard.replit.app (or connect to your local backend via ngrok)
2. Paste `https://github.com/FlashCarrot/VeriMedia` into the GitHub URL input
3. Click **Evaluate**
4. The three expert cards appear with color-coded status, risk, and findings
5. The final council verdict (ESCALATE banner) appears at the top
6. Click **View Full Report** to open the formatted HTML report

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | React 18 + Vite |
| API client | React Query v5 + custom fetch wrapper |
| Schema validation | Zod (generated from OpenAPI spec via orval v8) |
| DB (optional) | Drizzle ORM + SQLite (evaluation history) |
| Hosting | Replit (always-on deployment) |
| Package manager | pnpm workspaces |
