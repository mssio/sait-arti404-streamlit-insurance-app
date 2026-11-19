# Insurance Charges Prediction App

A Streamlit web app that predicts a patient's hospital insurance charges using a Gradient Boosting Regressor model trained with PyCaret.

## Features

- **Online prediction**: enter age, sex, BMI, number of children, smoker status and region to get a single prediction.
- **Batch prediction**: upload a CSV file and get predictions for every row.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.11. uv installs it automatically if it's missing. pycaret 3.3.2 doesn't support Python 3.12 or newer.

The model (`gbr_insurance_charges.pkl`) was trained with pandas 1.5.3, so pandas is pinned to that version in `pyproject.toml`. pandas 2.x can't load the model.

## Setup and run

```bash
uv sync
uv run streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Deployment

The app is deployed on Streamlit Community Cloud: https://insurance-mss.streamlit.app/

Community Cloud installs dependencies from `uv.lock`. When deploying, choose **Python 3.11** under **Advanced settings**. Pushes to `main` update the app automatically.

## Batch CSV format

The CSV needs these columns:

| Column     | Example     |
|------------|-------------|
| `age`      | `30`        |
| `sex`      | `male` / `female` |
| `bmi`      | `25.0`      |
| `children` | `0`         |
| `smoker`   | `yes` / `no` |
| `region`   | `southwest`, `northwest`, `northeast`, `southeast` |

## Project files

- `app.py`: the Streamlit app
- `gbr_insurance_charges.pkl`: the trained PyCaret model
- `logo.png`, `hospital.jpg`: images shown in the app
