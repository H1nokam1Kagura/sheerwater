"""Functions to fetch and process ECMWF IFS S2S reforecast data from Planette's Icechunk archive.

Planette hosts the ECMWF IFS-CY49R1 S2S reforecast archive (10-member ensemble, 46-day lead,
2006-2025) as a public, anonymous-access Icechunk/Zarr store on S3. See
https://github.com/PlanetteAI/planette_ifs_archive for details on the source dataset.
"""
import icechunk
import numpy as np
import xarray as xr
from nuthatch import cache
from nuthatch.processors import timeseries

from sheerwater.utils import dask_remote, densify_fcst, get_variable, lon_base_change
from sheerwater.interfaces import forecast as sheerwater_forecast

# Public, anonymous-access S3 bucket hosting the Planette IFS 46-day reforecast archive.
PLANETTE_BUCKET = "planette-ifs-46-day"
PLANETTE_PREFIX = "reforecasts"
PLANETTE_REGION = "us-east-2"

# Icechunk group for surface/atmosphere fields (t2m, precip, etc.) at 1.5x1.5 degree resolution.
# The archive also has "pressure/1p5latx1p5lon" (atmospheric fields on pressure levels) and
# "single/1platx1plon" (1x1 degree ocean fields), neither of which is fetched here.
PLANETTE_SURFACE_GROUP = "single/1p5latx1p5lon"


def _open_planette_group(group):
    """Open a group of the Planette IFS reforecast Icechunk store as a lazily-loaded xarray Dataset."""
    storage = icechunk.s3_storage(
        bucket=PLANETTE_BUCKET,
        prefix=PLANETTE_PREFIX,
        region=PLANETTE_REGION,
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    store = repo.readonly_session("main").store
    return xr.open_zarr(store, group=group, consolidated=False, chunks={})


@dask_remote
@timeseries(timeseries=['init_time'])
@cache(cache=False,
       cache_args=['variable', 'grid'],
       backend_kwargs={'chunking': {"lat": 121, "lon": 240, "prediction_timedelta": 46,
                                    "init_time": 200, "member": 10}})
def ifs_planette_reforecast_raw(start_time, end_time, variable='precip',  # noqa: ARG001
                                grid='global1_5', mask=None, region='global'):  # noqa: ARG001
    """Fetches raw ECMWF IFS S2S reforecast data from Planette's public Icechunk archive.

    This is intentionally left uncached: the underlying Zarr store is cheap to open lazily, we
    never need the raw data itself, and the only consumer is `clim_planette_ifs_er` below, whose
    output is what actually gets cached.

    Args:
        start_time (str): The start date to fetch data for (filters on `init_time`).
        end_time (str): The end date to fetch data for (filters on `init_time`).
        variable (str): The weather variable to fetch.
        grid (str): The grid resolution to fetch the data at. Only "global1_5" (the archive's
            native atmosphere/surface resolution) is currently implemented.
        mask: Unused; present for interface consistency.
        region: Unused; present for interface consistency.
    """
    if grid != 'global1_5':
        raise NotImplementedError(
            "Only the native 1.5 degree global grid is implemented for Planette reforecasts.")

    ds = _open_planette_group(PLANETTE_SURFACE_GROUP)

    var = get_variable(variable, 'planette')
    ds = ds[[var]]

    # Standardize dimension naming to sheerwater conventions
    ds = ds.rename({'number': 'member', 'lead': 'prediction_timedelta'})
    ds['prediction_timedelta'] = ds['prediction_timedelta'].astype('timedelta64[D]').astype('timedelta64[ns]')

    ds = ds.rename_vars(name_dict={var: variable})

    # Convert to base180 longitude
    ds = lon_base_change(ds, to_base="base180")

    # Unit conversions
    if variable in ['tmp2m', 'tmax2m', 'tmin2m', 'sst']:
        ds[variable] = ds[variable] - 273.15
        ds[variable].attrs.update(units='C')
    elif variable == 'precip':
        # Planette's raw field is GRIB paramId 228228 ("tprate"/"Daily mean precipitation
        # rate"), a rate in kg/m^2/s -- NOT an accumulated depth in meters (that would be
        # paramId 228, "tp"). Its GRIB_units attr misleadingly says "kg m**-2" (a known
        # cfgrib/eccodes artifact), but the decoded `units` attr ("kg m**-2 s**-1") and
        # long_name agree it's a rate. Converting to a daily mm total needs multiplying by
        # seconds/day (86400), not the *1000 meters->mm factor that applies to an
        # already-accumulated `tp`-style field.
        ds[variable] = ds[variable] * 86400.0
        ds[variable].attrs.update(units='mm / day')
        ds[variable] = np.maximum(ds[variable], 0)
    elif variable in ['uwind10m', 'vwind10m']:
        # No value conversion needed -- Planette's archive already provides these in m/s.
        ds[variable].attrs.update(units='m/s')

    # Rechunk to small spatial blocks with every time-like dim as a single chunk -- bfill/ffill
    # (in `densify_fcst`, downstream) and the dayofyear groupby-reduce are both fundamentally
    # per-grid-cell operations along time, so time/lead/member should never be split, while lat/lon
    # can be split as finely as needed to keep chunk byte size reasonable. `chunks={}` on open_zarr
    # above otherwise leaves us with the archive's native (small, write-optimized) Zarr chunks,
    # which blows up the task graph once anything downstream touches init_time/lead/member.
    # At N=3515 init_times x 46 leads x 10 members (float32), a 4x4 lat/lon chunk is ~103MB.
    ds = ds.chunk({'lat': 4, 'lon': 4, 'init_time': -1, 'prediction_timedelta': -1, 'member': -1})

    return ds


@dask_remote
@sheerwater_forecast()
@cache(cache=False,
       cache_args=['variable', 'agg_days', 'forecast_type', 'event', 'event_kwargs', 'processors',
                   'processor_kwargs', 'lookback_source', 'densify', 'prob_type', 'grid', 'mask', 'region'],
       backend_kwargs={'chunking': {'lat': 300, 'lon': 300, 'time': 365, 'lead_time': 1, 'member': 1}})
def ecmwf_ifs_planette_reforecast(start_time=None, end_time=None, variable="precip", agg_days=1,  # noqa: ARG001
                                  forecast_type='reforecast',
                                  prob_type='deterministic',
                                  event=None, event_kwargs=None,  # noqa: ARG001
                                  processors=None, processor_kwargs=None,  # noqa: ARG001
                                  lookback_source=None, densify=True,
                                  grid='global1_5', mask='lsm', region="global"):
    """Standard-format ECMWF IFS S2S reforecast data (10-member ensemble, 46-day lead) from Planette.

    Args:
        forecast_type (str): The type of forecast to fetch. Planette's archive only contains
            reforecast (hindcast) data, so only "reforecast" is supported; anything else raises.
        densify (bool): Whether to fill gaps between reforecast issuance dates. Planette only
            issues reforecasts a few times a week, so for a fixed lead, most calendar days have
            no init date at all; densifying bfills/ffills the nearest surrounding reforecast to
            cover those gaps. Defaults to True, unlike other forecast sources, since sparse
            issuance is the norm here rather than the exception. Note: this is handled explicitly
            here rather than through the standard `densify` mechanism on `@sheerwater_forecast()`,
            since that only densifies when an `event` is also requested.
    """
    if forecast_type != 'reforecast':
        raise NotImplementedError(
            "Planette's archive only contains reforecast data; forecast_type must be 'reforecast'.")

    ds = ifs_planette_reforecast_raw(start_time, end_time, variable=variable,
                                     grid=grid, mask=None, region='global')

    if prob_type == 'deterministic':
        ds = ds.mean(dim='member')
        ds = ds.assign_attrs(prob_type="deterministic")
    else:
        ds = ds.assign_attrs(prob_type="ensemble")

    if densify:
        attrs = ds.attrs
        ds = densify_fcst(ds, start_time=start_time, end_time=end_time)
        ds = ds.assign_attrs(attrs)

    if 'spatial_ref' in ds.variables:
        ds = ds.drop_vars('spatial_ref')

    return ds
