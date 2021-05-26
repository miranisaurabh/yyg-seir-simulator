Vaccination Effects:

The implementation of vaccination effects is mainly starts at Line 100 of simulation.py. Other than that, initializations, argument parser etc. are spread around all files.

12/13/2020: First vaccine doses administered in the US. Simulation matching real numbers at that date. However it doesn't quite match for earlier dates like 10/13 or 11/13.

- python run_simulation.py -v --best_params_dir best_params/latest --country US --simulation_end_date 2020-12-13 --skip_hospitalizations

05/25/2021: What would the numbers look like if there was no vaccination at all in the US? The simulator predicts extremely high numbers, probably a bit unrealistic.

- python run_simulation.py -v --best_params_dir best_params/latest --country US --simulation_end_date 2021-05-25 --skip_hospitalizations

05/25/2021: What would the numbers look like if we include the affects of vaccination???

- python run_simulation.py -v --best_params_dir best_params/latest --country US --simulation_end_date 2021-05-25 --skip_hospitalizations --include_vaccination