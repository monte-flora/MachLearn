import sys
sys.path.append('/home/monte.flora/machine_learning/build_model') 
from PreProcess import PreProcess 
from datetime import datetime

""" usage: stdbuf -oL python save_ml_data.py 2 > & log & """
model = PreProcess( )


#modes = [TRAINING, VALIDATION, TESTING]
modes = ['testing']

for key in ['first_hour', 'second_hour']:
    print('\n Start Time:', datetime.now().time())
    print("Forecast Time Index: ", key )
    model.save_data( key=key,
                     save_filename_template = '{}_f:{}_t:{}_raw_probability_objects.pkl',
                     var = 'PROB',
                     modes=modes
                     )
    print('End Time: ', datetime.now().time())

