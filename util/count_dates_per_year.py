import pandas as pd 

df = pd.read_pickle('/home/monte.flora/machine_learning/util/wofs_dates_for_verification.pkl')
ml_dates = df['Dates']

def count_num_of_years( df, year='2017'):
    return ml_dates.str.contains(year).sum()

count_2017 = count_num_of_years( df, year='2017')
count_2018 = count_num_of_years( df, year='2018')
count_2019 = count_num_of_years( df, year='2019')

print (f'Num of 2019 Dates: {count_2019} \n \
         Num of 2018 Dates: {count_2018} \n \
         Num of 2017 Dates: {count_2017}')






