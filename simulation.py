"""Underlying simulator for the YYG/C19Pro SEIR model.

Learn more at: https://github.com/youyanggu/yyg-seir-simulator. Developed by Youyang Gu.
"""

import datetime

import numpy as np
import pandas as pd

from fixed_params import *


def get_daily_imports(region_model, i):
    """Returns the number of new daily imported cases based on day index i (out of N days).

    - beginning_days_flat is how many days at the beginning we maintain a constant import.
    - end_days_offset is the number of days from the end of the projections
        before we get 0 new imports.
    - The number of daily imports is initially region_model.daily_imports, and
        decreases linearly until day N-end_days_offset.
    """

    N = region_model.N
    assert i < N, 'day index must be less than total days'

    if hasattr(region_model, 'beginning_days_flat'):
        beginning_days_flat = region_model.beginning_days_flat
    else:
        beginning_days_flat = 10
    assert beginning_days_flat >= 0

    if hasattr(region_model, 'end_days_offset'):
        end_days_offset = region_model.end_days_offset
    else:
        end_days_offset = int(N - min(N, DAYS_WITH_IMPORTS))
    assert beginning_days_flat + end_days_offset <= N
    n_ = N - beginning_days_flat - end_days_offset + 1

    daily_imports = region_model.daily_imports * \
        (1 - min(1, max(0, (i-beginning_days_flat+1)) / n_))

    if region_model.country_str not in ['China', 'South Korea', 'Australia'] and not \
            hasattr(region_model, 'end_days_offset'):
        # we want to maintain ~10 min daily imports a day
        daily_imports = max(daily_imports, min(10, 0.1 * region_model.daily_imports))

    return daily_imports


def run(region_model):
    """Given a RegionModel object, runs the SEIR simulation."""
    dates = np.array([region_model.first_date + datetime.timedelta(days=i) \
        for i in range(region_model.N)])
    infections = np.array([0.] * region_model.N)
    vaccinations_1 = np.array([0.] * region_model.N)
    vaccinations_2 = np.array([0.] * region_model.N)
    hospitalizations = np.zeros(region_model.N) * np.nan
    deaths = np.array([0.] * region_model.N)
    reported_deaths = np.array([0.] * region_model.N)
    mortaility_rates = np.array([region_model.MORTALITY_RATE] * region_model.N)

    assert infections.dtype == hospitalizations.dtype == \
        deaths.dtype == reported_deaths.dtype == mortaility_rates.dtype == np.float64

    """
    We compute a normalized version of the infections and deaths probability distribution.
    We invert the infections and deaths norm to simplify the convolutions we will take later.
        Aka the beginning of the array is the farther days out in the convolution.
    """
    deaths_norm = DEATHS_DAYS_ARR[::-1] / DEATHS_DAYS_ARR.sum()
    infections_norm = INFECTIOUS_DAYS_ARR[::-1] / INFECTIOUS_DAYS_ARR.sum()
    if hasattr(region_model, 'quarantine_fraction'):
        # reduce infections in the latter end of the infectious period, based on reduction_idx
        infections_norm[:region_model.reduction_idx] = \
            infections_norm[:region_model.reduction_idx] * (1 - region_model.quarantine_fraction)
        infections_norm[region_model.reduction_idx] = \
            (infections_norm[region_model.reduction_idx] * 0.5) + \
            (infections_norm[region_model.reduction_idx] * 0.5 * \
                (1 - region_model.quarantine_fraction))

    # the greater the immunity mult, the greater the effect of immunity
    assert 0 <= region_model.immunity_mult <= 2, region_model.immunity_mult
    
    vaccination_forecasts = pd.read_csv('vaccine_forecasts_both_dose.csv')

    ########################################
    # Compute infections
    ########################################
    effective_r_arr = []
    for i in range(region_model.N):
        if i < INCUBATION_DAYS+len(infections_norm):
            # initialize infections
            infections[i] = region_model.daily_imports
            effective_r_arr.append(region_model.R_0_ARR[i])
            continue

        # assume 50% of population lose immunity after 6 months
        # TODO: Use vaccine efficacy here?
        infected_thus_far = infections[:max(0, i-180)].sum() * 0.5 + infections[max(0, i-180):i-1].sum()
        perc_population_infected_thus_far = \
            min(1., infected_thus_far / region_model.population)
        assert 0 <= perc_population_infected_thus_far <= 1, perc_population_infected_thus_far

        if region_model.include_vaccination:
            vaccinated_1st_dose = vaccinations_1.sum()
            vaccinated_2nd_dose = vaccinations_2.sum()
            vaccinated_thus_far = int(DOSE1_EFFICACY*vaccinated_1st_dose + DOSE2_EFFICACY*vaccinated_2nd_dose)

            # We can include first dose / second dose here
            #vaccinated_thus_far = int(vaccinations_2.sum() * 0.9) + \
            #                      int((vaccinations_1.sum() - vaccinations_2.sum()) * 0.8) # 90% and 80% efficacy of doses
            # We can include the affect of vaccination decreasing after 8 months here
            # ...
            perc_population_vaccinated_thus_far = \
                min(1., vaccinated_thus_far / region_model.population)
                # min(1. - perc_population_infected_thus_far, vaccinated_thus_far / region_model.population)
            assert 0 <= perc_population_vaccinated_thus_far <= 1, perc_population_vaccinated_thus_far

            r_immunity_perc = (1. - perc_population_infected_thus_far - perc_population_vaccinated_thus_far)\
                              **region_model.immunity_mult # Does the immunity multiplier change?
            # r_immunity_perc = (1. - perc_population_infected_thus_far - perc_population_vaccinated_thus_far) \
            #                   ** region_model.immunity_mult  # Does the immunity multiplier change?
        else:
            r_immunity_perc = (1. - perc_population_infected_thus_far)**region_model.immunity_mult
        effective_r = region_model.R_0_ARR[i] * r_immunity_perc
        if effective_r <= 0 or np.isnan(effective_r):
            effective_r = 0.00001
        
        # we apply a convolution on the infections norm array
        s = (infections[i-INCUBATION_DAYS-len(infections_norm)+1:i-INCUBATION_DAYS+1] * \
            infections_norm).sum() * effective_r
        infections[i] = s + get_daily_imports(region_model, i)
        #######################
        if region_model.include_vaccination:
            if i > 310: # Vaccinations started 311 days after the simulation start date
                # Could make the rollout start date variable
                # Should our vaccine efficacy be arrays?
                # We probably want a dynamic vaccination rate, based on the CDC data
                #vaccinations_1[i] = region_model.population * 0.002
                #if i < region_model.N - 14: # 2 weeks between the doses
                #    vaccinations_2[i+14] = vaccinations_1[i] * 0.8 # 80% individuals get their second dose
                vaccinations_1[i] = vaccination_forecasts['Dose1'][i - 311] - vaccination_forecasts['Dose2'][i - 311]
                vaccinations_2[i] = vaccination_forecasts['Dose2'][i - 311]
            else:
                vaccinations_1[i] = 0
                vaccinations_2[i] = 0
    
        effective_r_arr.append(effective_r)

    region_model.perc_population_infected_final = perc_population_infected_thus_far
    assert len(region_model.R_0_ARR) == len(effective_r_arr) == region_model.N
    region_model.effective_r_arr = effective_r_arr

    ########################################
    # Compute hospitalizations
    ########################################
    if region_model.compute_hospitalizations:
        """
        Simple estimation of hospitalizations by taking the sum of a
            window of n days of new infections * hospitalization rate
        Note: this represents hospital beds used on on day _i, not new hospitalizations
        """
        for _i in range(region_model.N):
            start_idx = max(0, _i-DAYS_UNTIL_HOSPITALIZATION-DAYS_IN_HOSPITAL)
            end_idx = max(0, _i-DAYS_UNTIL_HOSPITALIZATION)
            hospitalizations[_i] = int(HOSPITALIZATION_RATE * infections[start_idx:end_idx].sum())

    ########################################
    # Compute true deaths
    ########################################
    assert len(deaths_norm) % 2 == 1, 'deaths arr must be odd length'
    deaths_offset = len(deaths_norm) // 2
    for _i in range(-deaths_offset, region_model.N-DAYS_BEFORE_DEATH):
        # we apply a convolution on the deaths norm array
        infections_subject_to_death = (infections[max(0, _i-deaths_offset):_i+deaths_offset+1] * \
            deaths_norm[:min(len(deaths_norm), deaths_offset+_i+1)]).sum()
        true_deaths = infections_subject_to_death * region_model.ifr_arr[_i + DAYS_BEFORE_DEATH]
        deaths[_i + DAYS_BEFORE_DEATH] = true_deaths

    ########################################
    # Compute reported deaths
    ########################################
    death_reporting_lag_arr_norm = region_model.get_reporting_delay_distribution()
    assert abs(death_reporting_lag_arr_norm.sum() - 1) < 1e-9, death_reporting_lag_arr_norm
    for i in range(region_model.N):
        """
        This section converts true deaths to reported deaths.

        We first assume that a small minority of deaths are undetected, and remove those.
        We then assume there is a reporting delay that is exponentially decreasing over time.
            The probability density function of the delay is encoded in death_reporting_lag_arr.
            In reality, reporting delays vary from region to region.
        """
        detected_deaths = deaths[i] * (1 - region_model.undetected_deaths_ratio_arr[i])
        max_idx = min(len(death_reporting_lag_arr_norm), len(deaths) - i)
        reported_deaths[i:i+max_idx] += \
            (death_reporting_lag_arr_norm * detected_deaths)[:max_idx]

    return dates, infections, hospitalizations, reported_deaths, [vaccinations_1, vaccinations_2]

