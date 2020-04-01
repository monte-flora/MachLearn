import multiprocessing as mp
import itertools
from wofs.evaluation.verification_metrics import Metrics
from os.path import join
import xarray as xr
from wofs.util import config
import numpy as np 

fcst_time_idx = 18
verify_var = 'matched_to_LSRs_30km'

forecasts = [ ]
targets = [ ]
for date, time in itertools.product( config.datetimes_ml, config.verification_times):    
    fname = join( join(config.ML_FCST_PATH, str(date)), f'WOFS_ML_OUTCOME_{date}_{time}_verify_var={verify_var}_fcst_time_idx={fcst_time_idx}.nc')
    ds = xr.open_dataset(fname)
    forecasts.extend(ds['Forecast'].values)
    targets.extend(ds['Targets'].values)

forecasts = np.array( forecasts )
targets = np.array( targets )

pod, sr, pofd = Metrics.performance_curve( forecasts, targets, bins=np.arange(0, 1.0, 0.05), roc_curve=True )
mean_fcst_probs, event_frequency = Metrics.reliability_curve( forecasts, targets, bins=np.arange(0, 1.0, 0.1))

data = { }
data['pod'] = (['thresholds'], pod)
data['sr'] = (['thresholds'], sr)
data['pofd'] = (['thresholds'], pofd)
data['mean fcst prob'] = (['bins'], mean_fcst_probs)
data['event frequency'] = (['bins'], event_frequency)

fname = join( config.ML_RESULTS_PATH, f'verifyData_WOFS_verify_var={verify_var}_fcst_time_idx={fcst_time_idx}.nc')
ds = xr.Dataset( data )
ds.to_netcdf(fname)
ds.close( )



