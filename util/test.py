import pickle


with open('operational_training_dates', 'rb') as fp:
    dates = pickle.load(fp)

print (len(dates))
