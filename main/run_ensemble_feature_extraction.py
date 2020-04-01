import os 
import joblib 
import itertools
import multiprocessing as mp
from multiprocessing.pool import Pool
import traceback
from ensemble_feature_extraction import worker 
from wofs.util import config
from tqdm import tqdm
from datetime import datetime
""" usage: stdbuf -oL python -u run_ensemble_feature_extraction.py 2 > & log & """

def error(msg, *args):
    """ Shortcut to multiprocessing's logger """
    return mp.get_logger().error(msg, *args)

class LogExceptions(object):
        def __init__(self, callable):
            self.__callable = callable

        def __call__(self, *args, **kwargs):
            try:
                result = self.__callable(*args, **kwargs)

            except Exception as e:
                # Here we add some debugging help. If multiprocessing's
                # debugging is on, it will arrange to log the traceback
                error(traceback.format_exc())
                # Re-raise the original exception so the Pool worker can
                # clean up
                raise

            # It was fine, give a normal answer
            return result

class LoggingPool(Pool):
    def apply_async(self, func, args=(), kwds={}, callback=None):
        return Pool.apply_async(self, LogExceptions(func), args, kwds, callback)

def to_iterator( *lists ):
    """
    turn list 
    """
    return itertools.product(*lists)

def update(*a):
    pbar.update()

def run_parallel( func, iterator, nprocs_to_use, mode='mp' ):
    '''
    Runs a series of python scripts in parallel

    Args:
    -------------------------
    func, python function, the function to be parallelized; can be a function which issues a series of python scripts 
    iterator, python iterator, the arguments of func to be iterated over
			     it can be the iterator itself or a series of list 

    nprocs_to_use, int or float, if int, taken as the literal number of processors to use
				if float (between 0 and 1), taken as the percentage of available processors to use

    '''
    
    if 0 <= nprocs_to_use < 1:
        nprocs_to_use = int(nprocs_to_use*mp.cpu_count())
    print(f'Using {nprocs_to_use} processors...') 

    if mode == 'joblib':
        backend = 'loky'
        joblib.Parallel(n_jobs = nprocs_to_use, 
                        backend=backend, 
                        verbose=100)(joblib.delayed(func)(*args) for args in iterator) 
    elif mode == 'mp':
        print("\n Start Time:", datetime.now().time())
        pool = LoggingPool(processes=nprocs_to_use)
        for args in iterator:
            pool.apply_async(func, args=args,callback=update)
        pool.close()
        pool.join()
        print("End Time: ", datetime.now().time())

dates = config.ml_dates
times = ['1900', '1930', '2000', '2030', '2100', '2130', '2200', '2230', '2300', '2330', '0000', '0030', '0100', '0130',
         '0200', '0230', '0300']
indexs = range(0, 24+1) 

pbar = tqdm( total =len(list(to_iterator(dates,times,indexs))))
run_parallel( 
	    func = worker,
	    nprocs_to_use = 0.3,
	    iterator = to_iterator(dates,times,indexs)
	    )


