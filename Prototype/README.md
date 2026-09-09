# AI Lab — Environment Lab UI

Frontend UI prototype for "Failure-Driven Environment Synthesis for Adaptive
AI Training". This is the **visualization/monitoring layer only** — mock
data, no backend, no real WebSocket/API calls yet. It's meant to be handed
to engineering as a clickable spec, or wired up to the real FastAPI backend
described in the project prompt.

## Stack

- Vite + React 18
- Tailwind CSS (utility classes only, dark theme via inline design tokens)
- recharts (charts)
- lucide-react (icons)

## Run locally

```bash
npm install
npm run dev
```

Then open http://localhost:5173

```bash
npm run build      # production build -> dist/
npm run preview    # preview the production build
```

## Folder structure

```
ai-lab-ui/
├── index.html                 # Vite HTML entry
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
├── src/
│   ├── main.jsx                # React root
│   ├── App.jsx                 # shell: sidebar + top bar + causal strip + page router
│   ├── index.css                # Tailwind directives
│   ├── theme.js                 # color tokens (dark research-lab palette)
│   ├── lib/
│   │   └── mockData.js          # deterministic RNG, gridworld generator, mock datasets
│   ├── components/
│   │   ├── ui.jsx               # Panel, Bar1, Badge, StatusDot, KV, SectionLabel
│   │   ├── GridWorld.jsx        # full grid renderer + MiniGrid thumbnail
│   │   ├── charts.jsx           # SuccessProgressChart, ComparisonChart, GeneralizationChart
│   │   ├── Sidebar.jsx          # nav + system status widget
│   │   ├── TopBar.jsx           # experiment meta bar + pause/stop controls
│   │   └── CausalStrip.jsx      # Agent → Failure → Weakness → Env → Improvement banner
│   └── pages/
│       ├── Overview.jsx         # Lab overview / dashboard
│       ├── Experiments.jsx      # experiment list
│       ├── CreateExperiment.jsx # new experiment form
│       ├── LiveTraining.jsx     # live grid-world + real-time stats
│       ├── Environments.jsx     # candidate environment generator
│       ├── Failures.jsx         # failure analysis
│       ├── Curriculum.jsx       # curriculum timeline + history table
│       ├── Evaluation.jsx       # generalization (train/val/test/OOD)
│       ├── Comparison.jsx       # Random vs Fixed vs Generic vs Failure-driven
│       └── Details.jsx          # experiment detail w/ tabs
└── README.md
```

## Wiring to the real backend later

Every page currently imports its data from `src/lib/mockData.js`. To connect
to the FastAPI backend from the master prompt, replace those imports with
calls to a `src/services/api.js` client (e.g. `fetch`/`axios` hitting
`/api/experiments/...`) and a `src/hooks/` folder for data-fetching hooks —
mirroring the `frontend/services` and `frontend/hooks` layout in the
original architecture doc. Live training numbers are the ones to wire to a
WebSocket first, since that's the one true real-time page.
