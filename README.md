# Groundnut Fungal Disease Forewarning System

Streamlit application for district-specific weather-based forecasting of:

- Groundnut Rust (*Puccinia arachidis*)
- Late Leaf Spot (Tikka disease)

## Districts

### Coimbatore
Uses the supplied 10-year weather + disease datasets.

### Cuddalore
Uses the supplied Cuddalore weather + disease dataset.

The Cuddalore disease data contains both Rust and Late Leaf Spot scores. Rabi/Zaid rows without disease observations are not used for Cuddalore model training.

## Forecasts

Each disease has three separate models:

1. Current Week: weather(t) -> disease(t)
2. Week 1: weather(t) -> disease(t+1)
3. Week 2: weather(t) -> disease(t+2)

That gives 12 district/disease/horizon models:

- Coimbatore Rust: Current, Week 1, Week 2
- Coimbatore Late Leaf Spot: Current, Week 1, Week 2
- Cuddalore Rust: Current, Week 1, Week 2
- Cuddalore Late Leaf Spot: Current, Week 1, Week 2

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub / Streamlit Cloud

1. Create a GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `data/`
   - `assets/`
3. Deploy the repository using Streamlit Community Cloud.
4. Set the main file to `app.py`.

## Data files

The project currently expects:

```text
data/
├── coimbatore_rust.xlsx
├── coimbatore_lls.xlsx
└── cuddalore.xlsx
```

## Important scientific note

The application trains linear regression equations automatically from the supplied historical data. The equations shown in the application are model-estimated coefficients, not manually invented coefficients.

The current risk bands are provisional, data-derived quartile bands. Before publication or operational agricultural recommendation, they should be replaced/confirmed using the disease-rating methodology and agronomic validation for each district and disease.


## Suggested GitHub structure

```text
groundnut-disease-forewarning/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── coimbatore_rust.xlsx
│   ├── coimbatore_lls.xlsx
│   └── cuddalore.xlsx
│
└── assets/
    ├── rust_symptom.jpg
    └── lls_symptom.jpg
```
