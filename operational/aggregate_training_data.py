import sys
sys.path.append('/home/monte.flora/machine_learning/build_model')
from PreProcess import PreProcess

model = PreProcess( )

for key in ['first_hour', 'second_hour']:
    model._save_operational_dataset( key )



