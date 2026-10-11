Getting started
===============

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make requirements
make pipeline     # download -> clean -> features -> models -> figures
make dashboard    # Streamlit at http://localhost:8501
```

Pipeline stages (each is also a standalone command):

| Command | Output |
|---|---|
| `python -m us_health_equity.dataset download` | `data/raw/` – cached CDC, WONDER, CMS and FRED downloads |
| `python -m us_health_equity.dataset process` | `data/interim/` – one tidy table per source |
| `python -m us_health_equity.features` | `data/processed/` – analysis tables read by the dashboard |
| `python -m us_health_equity.modeling.train` | `models/` – state-level associations, spending trend |
| `python -m us_health_equity.plots` | `reports/figures/` – static figures |

CDC WONDER requires 15 seconds between API calls, so the first download takes a few minutes.
No API keys are required. See the README for sources, methods and caveats.
