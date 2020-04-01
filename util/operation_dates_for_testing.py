import pandas as pd
import random
import pickle

df = pd.read_pickle('wofs_dates_for_verification.pkl')

dates = df['Dates'].values.astype(str)
random.shuffle(dates)

n_training_dates = int(0.8*len(dates))

training_dates = dates[:n_training_dates]
validation_dates = dates[n_training_dates:]

with open('operational_training_dates', 'wb') as pkl_file:
    pickle.dump(training_dates, pkl_file)

with open('operational_validation_dates', 'wb') as pkl_file:
    pickle.dump(validation_dates, pkl_file)


