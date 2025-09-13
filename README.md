# Q‑Route+ Visualizer

A lightweight, browser‑only demo of risk‑aware vehicle routing with stochastic demands. It showcases the Quantile Route Oracle (QRO) capacity check, optional correlation between site demands, greedy construction with 2‑opt improvement, and Monte Carlo validation with charts — all in a single static HTML file.

## Highlights
- Zero build. Open in a browser — React, Tailwind, Babel, and Chart.js are loaded via CDN.
- Risk‑aware capacity: Normal quantile or Bernstein bound with adjustable ε (risk) and b (radius).
- Correlation slider ρ ∈ [0, 0.9]: approximate pairwise covariance between site demands.
- Heuristics: Greedy insertion (QRO‑feasible) + optional 2‑opt local search.
- Step‑through demo: Watch one insertion at a time.
- Monte Carlo analysis: Per‑route overflow rates, scatter calibration, and histograms.
- Transparent data: “SHOW DATA” modal with generated instance details.

## Quick start
- Option A: Double‑click `index_fixed.html` to open it in your default browser.
- Option B (recommended for GitHub Pages parity): Serve files over HTTP.
  - VS Code extension “Live Server”, or on Windows PowerShell:
    ```powershell
    # If you have Python installed
    python -m http.server 8080
    # Then open http://localhost:8080/index_fixed.html
    ```

Note: `index.html` is a simple redirect to `index_fixed.html` so GitHub Pages (root) loads the stable app.

## Controls & KPIs
- Instance: number of sites, vehicles, capacity (L), speed (km/h), ε (risk), emissions factor, time windows on/off, correlation ρ, and bound type (Bernstein / Normal).
- Run buttons:
  - Run (Greedy): builds QRO‑feasible routes.
  - Run + 2‑opt: improves route distance while preserving feasibility.
  - Start Step / Step Once: stepwise visualization of greedy insertions.
  - Simulate Overflow: runs a Monte Carlo and opens charts.
- KPIs dashboard includes distance, drive time, CO₂, skipped sites, capacity and vehicle utilization, risk ratios, time window violations, cost breakdown, and route balance.

## Algorithms inside
- Quantile Route Oracle (QRO):
  - Normal bound: μ + z_{1−ε} √Var
  - Bernstein bound: μ + √(2 Var ln(1/ε)) + (2/3) b ln(1/ε)
- Correlation model (demo): global ρ adds cross‑terms Cov(i,j) ≈ ρ σ_i σ_j.
- Greedy insertion: tries all vehicle positions that respect time windows and QRO; picks minimal added distance.
- 2‑opt: opportunistic segment reversal if time + QRO remain feasible and distance reduces.

## Monte Carlo analytics
- Simulates realized volumes per route using Normal(μ, σ) draws.
- Reports overall overflow frequency and per‑route rates.
- Charts (Chart.js):
  - Scatter: predicted risk ratio (threshold/capacity) vs realized overflow.
  - Histogram: load distribution for a selected route.

## Project structure
- `index_fixed.html` — Main single‑file React app (stable entry; includes Monte Carlo charts).
- `index.html` — Redirects to `index_fixed.html` (useful for GitHub Pages root).
- `code/q_route_web_simulation_react.jsx` — Alternative, richer JSX prototype (not required to run the stable app).
- `code/main_kpis.ipynb` — Notebook used for KPI exploration.
- `code/flowchart.*`, `code/problem_overview.jpg` — Visuals and diagrams.
- `paper/` — Reference PDFs for background and related work.

## Developing / modifying
- The app uses in‑browser Babel for JSX (no build step). Edit `index_fixed.html` and refresh.
- Major sections to explore:
  - Data generation (sites/vehicles, time windows)
  - QRO bounds and correlation handling
  - Greedy insertion and `twoOptImprove`
  - Monte Carlo: `simulateOverflow` and chart modal
- You can lift the logic into a modern React build (Vite/CRA/Next) later; this demo keeps friction low.

## Deploy to GitHub Pages
1. Commit the repository to GitHub.
2. In your repo settings → Pages → set Source to your default branch, root (`/`).
3. Ensure `index.html` exists (it already redirects to `index_fixed.html`).
4. Open the GitHub Pages URL; the app should load.

## Troubleshooting
- Blank page or CORS warnings: open via HTTP (Live Server or `python -m http.server`) instead of `file://`.
- CDNs blocked offline: connect to the internet or bundle dependencies locally.
- Slow charts: reduce Monte Carlo trials in the UI or code.

## License
Add your preferred license (e.g., MIT) as `LICENSE` in the repo root.

## Citation
If you use this demo in research, please cite your accompanying paper or repository. Replace this section with your canonical citation once finalized.
