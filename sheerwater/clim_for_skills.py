"""Simplified daily climatologies (mean + std) for skills.
"""

import numpy as np
import xarray as xr
from nuthatch import cache

from sheerwater.interfaces import spatial, get_data, get_forecast
from sheerwater.utils import (
    add_dayofyear,
    convert_pred_time_to_init_time,
    dask_remote,
    get_dates,
    pad_with_leapdays,
)


@dask_remote
@spatial()
@cache(cache=False,
       cache_args=['data_name', 'variable', 'first_year', 'last_year', 'grid', 'mask', 'region'],
       backend_kwargs={
           'chunking': {"lat": 121, "lon": 240, "dayofyear": 1000, "prediction_timedelta": 1},
           'chunk_by_arg': {
               'grid': {
                   'global0_25': {"lat": 721, "lon": 1440, 'dayofyear': 30, 'prediction_timedelta': 1}
               }
           }
})
def clim_stats_raw(data_name, variable, first_year=1985, last_year=2014, is_forecast=False,
                    grid='global1_5', mask=None, region='global'):
    """Compute the per-day-of-year climatological mean and standard deviation for a data source.

    Returns:
        xr.Dataset: Dataset with a `dayofyear` dimension (and `prediction_timedelta`,
            if `is_forecast=True`) containing `avg` and `std`.
    """
    data_fn = get_forecast(data_name) if is_forecast else get_data(data_name)

    start_time = f"{first_year}-01-01"
    end_time = f"{last_year}-12-31"
    ds = data_fn(start_time, end_time, variable=variable, agg_days=1, grid=grid, mask=mask, region=region)
    ds = ds[[variable]]

    # Capture units before the groupby-reduce below drops attrs; check both per-variable and
    # dataset-level attrs since sources use either convention.
    units = ds[variable].attrs.get('units') or ds.attrs.get('units')

    try:
        ds = add_dayofyear(ds)
        ds = pad_with_leapdays(ds)
    except KeyError:
        raise ValueError(f"Underlying data source {data_name} does not have full coverage "
                         f"of the climatology period {first_year}-{last_year}.")

    # Eager labels needed for the groupby below once `ds` is dask-backed/chunked.
    dayofyear_labels = np.unique(ds['dayofyear'].values)

    # NOTE: rechunk-before-groupby disabled while testing Planette (its own fetch already
    # rechunks time-like dims to single chunks). Re-enable for era5/imerg/chirps/ecmwf_ifs_er:
    # rechunk = {'lat': 48, 'lon': 48, 'time': -1}
    # if 'prediction_timedelta' in ds.dims:
    #     rechunk['prediction_timedelta'] = -1
    # ds = ds.chunk(rechunk)

    grouped = ds.groupby(dayofyear=xr.groupers.UniqueGrouper(labels=dayofyear_labels))
    mean_ds = grouped.mean(dim='time', skipna=True).rename({variable: 'avg'})
    std_ds = grouped.std(dim='time', skipna=True).rename({variable: 'std'})
    ds = xr.merge([mean_ds, std_ds])

    # std shares the mean's units (unlike variance, which would be squared).
    if units is not None:
        ds['avg'].attrs['units'] = units
        ds['std'].attrs['units'] = units

    return ds


@dask_remote
@cache(cache=True,
       cache_args=['variable', 'data_name', 'first_year', 'last_year',
                   'forecast_lead_days', 'prob_type', 'grid', 'mask', 'region'],
       backend_kwargs={'chunking': {'lat': 300, 'lon': 300, 'init_time': 365, 'prediction_timedelta': 1}})
def clim_daily(start_time, end_time, variable, data_name, agg_days=1,  # noqa: ARG001
                first_year=1985, last_year=2014, is_forecast=False,
                forecast_lead_days=None, prob_type='deterministic',  # noqa: ARG001
                grid='global0_25', mask=None, region='global'):
    """Standard-format daily climatology (mean + std) for a given data source.
    """
    raw = clim_stats_raw(data_name, variable, first_year=first_year, last_year=last_year,
                         is_forecast=is_forecast, grid=grid, mask=mask, region=region)

    target_dates = get_dates(start_time, end_time, stride='day', return_string=False)
    time_ds = add_dayofyear(xr.Dataset({'time': target_dates}))

    ds = raw.sel(dayofyear=time_ds.dayofyear)
    ds = ds.drop_vars('dayofyear')
    ds = ds.assign_attrs(prob_type="deterministic")

    if not is_forecast:
        # climatology has no lead-dependence -- broadcast the same value across synthetic leads.
        leads = [np.timedelta64(x, "D").astype('timedelta64[ns]') for x in range(0, forecast_lead_days)]
        ds = ds.expand_dims({"prediction_timedelta": leads})

    ds = convert_pred_time_to_init_time(ds)
    ds = ds.sel(init_time=slice(start_time, end_time))
    ds = ds.transpose('lat', 'lon', 'init_time', 'prediction_timedelta')
    ds = ds.chunk({'lat': 121, 'lon': 240, 'prediction_timedelta': 1})
    return ds


@dask_remote
@cache(cache=False,
       cache_args=['variable', 'agg_days', 'prob_type', 'grid', 'mask', 'region'],
       backend_kwargs={'chunking': {'lat': 300, 'lon': 300, 'init_time': 365, 'prediction_timedelta': 1}})
def clim_era5(variable, agg_days=1,  # noqa: ARG001
              prob_type='deterministic',  # noqa: ARG001
              grid='global0_25', mask=None, region='global'):
    """Daily ERA5 climatology (mean + std), 1990-2019.
    """
    return clim_daily('1904-01-01', '1904-12-31', variable, data_name='era5',
                      first_year=1990, last_year=2019, is_forecast=False,
                      forecast_lead_days=1, grid=grid, mask=mask, region=region)


@dask_remote
@cache(cache=False,
       cache_args=['variable', 'agg_days', 'prob_type', 'grid', 'mask', 'region'],
       backend_kwargs={'chunking': {'lat': 300, 'lon': 300, 'init_time': 365, 'prediction_timedelta': 1}})
def clim_imerg(variable, agg_days=1,  # noqa: ARG001
               prob_type='deterministic',  # noqa: ARG001
               grid='global0_25', mask=None, region='global'):
    """Daily IMERG climatology (mean + std), 1998-2024.
    """
    return clim_daily('1904-01-01', '1904-12-31', variable, data_name='imerg_final',
                      first_year=1998, last_year=2024, is_forecast=False,
                      forecast_lead_days=1, grid=grid, mask=mask, region=region)


@dask_remote
@cache(cache=False,
       cache_args=['variable', 'agg_days', 'prob_type', 'grid', 'mask', 'region'],
       backend_kwargs={'chunking': {'lat': 300, 'lon': 300, 'init_time': 365, 'prediction_timedelta': 1}})
def clim_chirps(variable, agg_days=1,  # noqa: ARG001
                prob_type='deterministic',  # noqa: ARG001
                grid='global0_25', mask=None, region='global'):
    """Daily CHIRPS v3 climatology (mean + std), 1998-2024.
    """
    return clim_daily('1904-01-01', '1904-12-31', variable, data_name='chirps_v3',
                      first_year=1998, last_year=2024, is_forecast=False,
                      forecast_lead_days=1, grid=grid, mask=mask, region=region)


@dask_remote
@cache(cache=False,
       cache_args=['variable', 'agg_days', 'prob_type', 'grid', 'mask', 'region'],
       backend_kwargs={'chunking': {'lat': 300, 'lon': 300, 'init_time': 365, 'prediction_timedelta': 1}})
def clim_oisst(variable, agg_days=1,  # noqa: ARG001
               prob_type='deterministic',  # noqa: ARG001
               grid='global0_25', mask=None, region='global'):
    """Daily NOAA OISST v2.1 climatology (mean + std), 1991-2020 (WMO 30-year normal period).

    `mask` defaults to None, not 'lsm' -- an 'lsm' mask keeps land and drops ocean, which is
    backwards for SST (OISST's own data is already only defined over ocean; land is NaN).
    """
    return clim_daily('1904-01-01', '1904-12-31', variable, data_name='oisst',
                      first_year=1991, last_year=2020, is_forecast=False,
                      forecast_lead_days=1, grid=grid, mask=mask, region=region)


@dask_remote
@cache(cache=False,
       cache_args=['variable', 'agg_days', 'prob_type', 'grid', 'mask', 'region'],
       backend_kwargs={'chunking': {'lat': 300, 'lon': 300, 'init_time': 365, 'prediction_timedelta': 1}})
def clim_ecmwf_ifs_er(variable, agg_days=1,  # noqa: ARG001
                      prob_type='deterministic',  # noqa: ARG001
                      grid='global0_25', mask=None, region='global'):
    """Daily, per-lead ECMWF IFS extended-range climatology (mean + std), 2018-2022.
    """
    return clim_daily('1904-01-01', '1904-12-31', variable, data_name='ecmwf_ifs_er',
                      first_year=2018, last_year=2022, is_forecast=True,
                      grid=grid, mask=mask, region=region)


@dask_remote
@cache(cache=False,
       cache_args=['variable', 'agg_days', 'prob_type', 'grid', 'mask', 'region'],
       backend_kwargs={'chunking': {'lat': 300, 'lon': 300, 'init_time': 365, 'prediction_timedelta': 1}})
def clim_ecmwf_ifs_planette_reforecast(variable, agg_days=1,  # noqa: ARG001
                                       prob_type='deterministic',  # noqa: ARG001
                                       grid='global1_5', mask='lsm', region='global'):
    """Daily, per-lead climatology (mean + std) for Planette's ECMWF IFS S2S reforecast archive, 2006-2024.

    Note: Planette's archive only has data on the native "global1_5" grid (see
    `ecmwf_ifs_planette_reforecast`), so `grid` must be "global1_5".
    """
    return clim_daily('1904-01-01', '1904-12-31', variable, data_name='ecmwf_ifs_planette_reforecast',
                      first_year=2006, last_year=2024, is_forecast=True,
                      grid=grid, mask=mask, region=region)
