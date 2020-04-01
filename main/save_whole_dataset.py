import sys
sys.path.append('/home/monte.flora/machine_learning/build_model')
from PreProcess import PreProcess

model = PreProcess( )
model._save_wholedataset()
