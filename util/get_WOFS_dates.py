import numpy as np 
import pandas as pd 
from glob import glob
from os.path import join
import os 

base_paths = [ '/scratch/skinnerp/2019_wofs_post/summary_files',
               '/oldscratch/skinnerp/2018_newse_post/summary_files']

good_dates = [ ]
for in_path in base_paths:
    potential_dates = [ ]
    dates = [ x.split('/')[-1] for x in glob( join(in_path, '20*')) ]
    potential_dates.extend( dates ) 
 
    for date in potential_dates:
        if len(os.listdir(join( join(in_path, date),'0000')))>0:
            good_dates.append( date.split('_')[0] )

good_dates.remove( '20170825' ) # Not Spring
good_dates.remove( '20170910' ) # Not spring 
good_dates.remove( '20180913' )
good_dates.remove( '20180914' )
good_dates.remove( '20180430' ) # Testing date 
good_dates.remove( '20190622' )
good_dates.remove( '20190621' )

# Remove FFAIR and summertime dates
good_months = [4,5,6]
spring_dates = [ ] 
for date in good_dates:
    if int(date[4:6]) in good_months:
        spring_dates.append(date)

print (sorted(spring_dates))    
print ('Num of Dates: {}'.format(len(spring_dates)))

data = {'Dates': spring_dates} 
df = pd.DataFrame( data )
df.to_pickle('wofs_dates_for_verification.pkl')


