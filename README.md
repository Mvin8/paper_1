# Urbanomy Paper

Minimal repository for `paper.ipynb`.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
jupyter notebook paper.ipynb
```

Set `OPENAI_API_KEY` in `.env` before running LLM cells.

## Contents

- `paper.ipynb` - main notebook
- `paper_data/` - baseline GeoJSON and saved paper data
- `data/catboost_land_value_no_services.cbm` - CatBoost model used by the notebook
- `nsga_2_without_llm/` and root `*.jsonl` files - saved optimization outputs used by analysis cells
- `src/urbanomy/` - minimal copied library components used by the notebook

`pareto_front.jsonl` is not included because it was not present in the source tree; rerun the notebook cell with `build_pareto_front_dataframe(...)` to regenerate it.
