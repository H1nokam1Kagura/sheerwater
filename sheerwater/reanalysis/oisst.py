"""Fetches NOAA OISST v2.1 daily sea-surface temperature from NOAA PSL's public OPeNDAP server."""
import dask
import xarray as xr
from dateutil import parser
from nuthatch import cache
from nuthatch.processors import timeseries

from sheerwater.interfaces import data as sheerwater_data, spatial
from sheerwater.utils import dask_remote, get_grid, get_variable, lon_base_change, regrid

# NOAA PSL serves OISST v2.1 as one NetCDF file per year via OPeNDAP; native resolution is
# 0.25x0.25 degree, daily, from 1981-09 to present.
OISST_OPENDAP_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"

# Max days per single OPeNDAP request
_MAX_DAYS_PER_REQUEST = 3


def _fetch_oisst_year(year, var):
    """Fetch one year of a raw OISST variable."""
    try:
        dy = xr.open_dataset(OISST_OPENDAP_URL.format(year=year))
    except Exception as exc:
        raise ValueError(
            f"Could not open the OISST file for {year} ({exc})."
        ) from exc
    with dy:
        sub = dy[[var]]
        n_days = sub.sizes.get('time', 0)
        pieces = [_load_batch_with_retry(sub, var, i, i + _MAX_DAYS_PER_REQUEST, year)
                 for i in range(0, n_days, _MAX_DAYS_PER_REQUEST)]
        ds = xr.concat(pieces, dim='time')
    return ds


def _load_batch_with_retry(sub, var, start, stop, year, max_attempts=3):
    """Load one time-batch, retrying if the server returns all-zero data.
    """
    for attempt in range(1, max_attempts + 1):
        piece = sub.isel(time=slice(start, stop)).load()
        if not bool((piece[var] == 0).all()):
            return piece
        if attempt < max_attempts:
            print(f"OISST {year} batch [{start}:{stop}] came back all-zero "
                 f"(attempt {attempt}/{max_attempts}), retrying...")
    raise ValueError(
        f"OISST {year} batch [{start}:{stop}] returned all-zero data on all {max_attempts} "
        "attempt.")


@dask_remote
@cache(cache_args=['year', 'variable', 'grid'],
       backend_kwargs={'chunking': {"lat": 300, "lon": 300, "time": 366}})
def oisst_daily_year(year, variable='sst', grid='global0_25', mask=None, region='global'):  # noqa: ARG001
    """Fetches one year of NOAA OISST v2.1 daily SST.

    Args:
        year (int): The year to fetch.
        variable (str): Only 'sst' is available from OISST.
        grid (str): Only OISST's native "global0_25" grid is implemented here.
    """
    if grid != 'global0_25':
        raise NotImplementedError("only supports native global0_25 grid.")

    var = get_variable(variable, 'oisst')
    ds = _fetch_oisst_year(year, var)

    # NOAA's raw file's `missing_value` attr survives into `.encoding` through the
    # open/concat chain, while xarray's own NaN-decoding independently sets `_FillValue`
    # there too -- the two conflict (different values) when writing to zarr. Clear it and
    # let zarr use its own default encoding instead.
    ds[var].encoding = {}

    # OISST is served in base360 longitude.
    ds = lon_base_change(ds, to_base="base180")
    ds = ds.rename_vars(name_dict={var: variable})

    # stamp units
    if variable == 'sst':
        ds[variable].attrs.update(units='C')

    ds = ds.chunk({'lat': 300, 'lon': 300, 'time': 366})
    return ds


@dask_remote
@timeseries()
@spatial()
@cache(cache_args=['variable', 'grid'],
       backend_kwargs={'chunking': {"lat": 48, "lon": 48, "time": -1}})
def oisst_daily(start_time, end_time, variable='sst', grid='global0_25', mask=None, region='global'):
    """NOAA OISST daily SST over a date range, fetching one year at a time in parallel.
    """
    if grid != 'global0_25':
        raise NotImplementedError("only supports native global0_25 grid.")

    years = range(parser.parse(start_time).year, parser.parse(end_time).year + 1)

    delayed_years = [dask.delayed(oisst_daily_year)(year, variable=variable, grid=grid,
                                                     mask=mask, region=region, filepath_only=True)
                     for year in years]
    paths = dask.compute(*delayed_years)

    ds = xr.open_mfdataset(paths, engine='zarr', parallel=True,
                          chunks={'lat': 300, 'lon': 300, 'time': 365})
    ds = ds.sel(time=slice(start_time, end_time))
    # clear encoding
    ds[variable].encoding = {}

    return ds


@dask_remote
@spatial()
@cache(cache_args=['variable', 'grid'],
       cache_disable_if={'grid': 'global0_25'},
       backend_kwargs={'chunking': {"lat": 48, "lon": 48, "time": -1}})
def oisst_daily_regrid(start_time, end_time, variable='sst', grid='global0_25', mask=None, region='global'):
    """NOAA OISST daily SST, regridded onto the requested grid."""
    ds = oisst_daily(start_time, end_time, variable=variable, grid='global0_25', mask=mask, region=region)

    if grid == 'global0_25':
        return ds

    _, _, size, _ = get_grid(grid)
    if size < 0.25:
        raise NotImplementedError("Unable to regrid OISST smaller than its native 0.25x0.25 grid.")

    ds = ds.chunk({'lat': 300, 'lon': 300, 'time': 365})
    ds = regrid(ds, grid, base='base180', method='conservative', region=region)
    return ds


@dask_remote
@sheerwater_data()
@cache(cache=False,
       cache_args=['variable', 'agg_days', 'event', 'event_kwargs', 'processors', 'processor_kwargs',
                   'grid', 'mask', 'region'],
       backend_kwargs={'chunking': {'lat': 300, 'lon': 300, 'time': 365}})
def oisst(start_time=None, end_time=None, variable='sst', agg_days=1,  # noqa: ARG001
         event=None, event_kwargs=None,  # noqa: ARG001
         processors=None, processor_kwargs=None,  # noqa: ARG001
         grid='global0_25', mask='lsm', region='global'):
    """Standard format task data for NOAA OISST v2.1 daily sea-surface temperature."""
    if variable != 'sst':
        raise NotImplementedError("Only 'sst' is available from OISST.")

    _, _, size, _ = get_grid(grid)
    if size < 0.25:
        raise NotImplementedError("Unable to regrid OISST smaller than its native 0.25x0.25 grid.")

    ds = oisst_daily_regrid(start_time, end_time, variable=variable, grid=grid, mask=mask, region=region)
    return ds
