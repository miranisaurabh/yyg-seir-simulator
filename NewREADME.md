# COVID-19 Projections Simulator

## Vaccine Predictions
### Usage
```python
from vaccine_preds import VaccinePredictor

# Create a VaccinePredictor object
PredObj = VaccinePredictor()
# Create a .csv file with forecasts by fitting a Gaussian curve
## Start Date: 12/13/2020
## End Date: Start Date + pred_ahead (default 120 days)
PredObj.write_forecast_csv(pred_ahead=120)
```