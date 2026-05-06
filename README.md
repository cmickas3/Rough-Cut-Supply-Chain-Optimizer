# Supply Plan Optimizer

This workspace now has a first Streamlit version of the notebook model.

## Files

- `CSV Based Project.ipynb`: working copy of the original notebook
- `supply_optimizer.py`: reusable optimization logic extracted from the notebook
- `app.py`: interactive Streamlit UI
- `requirements.txt`: Python dependencies for the app

## Run It

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

The app accepts the two CSVs from the notebook:

- `Project Mock Data - Demand and NIT (2).csv`
- `Project Mock Data - Capacity (7).csv`

This workspace includes example data and the app will load it automatically:

- `Project Mock Data - Demand and NIT (1).csv`
- `Project Mock Data - Capacity (3).csv`

You can also upload another demand/capacity pair in the sidebar.
