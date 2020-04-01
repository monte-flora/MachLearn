import pandas as pd 
import os
from wofs.util import config
import numpy as np 

fold = 0; fcst_time_idx=6; 
data_extraction_method = 'hagelslag'
train_df = pd.read_pickle( os.path.join(config.ML_DATA_STORAGE_PATH, f'training_dataset_fold={fold}_fcst_time_idx={fcst_time_idx}_{data_extraction_method}.pkl' ))

variables_to_remove = ['MRMS Reflectivity @ 0 min_mean', 'MRMS Reflectivity @ 15 min_mean', 'MRMS Reflectivity @ 30 min_mean',
                    'matched_to_severe_wx_warn_polys_15km', 'matched_to_severe_wx_warn_polys_30km',
                    'matched_to_tornado_warn_ploys_15km', 'matched_to_tornado_warn_ploys_30km']

variables_to_remove += ['matched_to_LSRs_30km', 'matched_to_azshr_30km', 'matched_to_Tornado_LSRs_15km', 'matched_to_Tornado_LSRs_30km',
                                'matched_to_Severe_Wind_LSRs_15km', 'matched_to_Severe_Wind_LSRs_30km',
                                'matched_to_Hail_LSRs_15km', 'matched_to_Hail_LSRs_30km', 'Run Time', 'Run Date', 'label', 'ensemble_member']

train_df = train_df.drop( columns = variables_to_remove)


def filter_df_by_corr(self, inp_data, target_val_name, cc_val=0.8):
        '''
        Returns an array or dataframe (based on type(inp_data) adjusted to drop \
            columns with high correlation to one another. Takes second arg corr_val
            that defines the cutoff

        ----------
        inp_data : np.array, pd.DataFrame
            Values to consider
        corr_val : float
            Value [0, 1] on which to base the correlation cutoff
        '''
        # Creates Correlation Matrix
        if isinstance(inp_data, np.ndarray):
            inp_data = pd.DataFrame(data=inp_data)
            array_flag = True
        else:
            array_flag = False
        corr_matrix = inp_data.corr()

        # Iterates through Correlation Matrix Table to find correlated columns
        drop_cols = []
        n_cols = len(corr_matrix.columns)

        print ('Calculating correlations between features...')
        for i in range(n_cols):
            for k in range(i+1, n_cols):
                val = corr_matrix.iloc[k, i]
                col = corr_matrix.columns[i]
                row = corr_matrix.index[k]
                col_to_target = corr_matrix.loc[col,target_val_name]
                row_to_target = corr_matrix.loc[row,target_val_name]
                if abs(val) >= cc_val: 
                    # Prints the correlated feature set and the corr val
                    if col_to_target > row_to_target and row not in drop_cols:
                        print( f'{col} ({col_to_target:.3f}) | {row} ({row_to_target:.3f}) | {val:.2f}....Dropped {row}')
                        drop_cols.append(row)
                    if row_to_target > col_to_target and col not in drop_cols:
                        print( f'{col} ({col_to_target:.3f}) | {row} ({row_to_target:.3f}) | {val:.2f}....Dropped {col}')
                        drop_cols.append(col)

        # Drops the correlated columns
        # Drops the correlated columns
        print ('Dropping {} highly correlated features...'.format(len(drop_cols)))
        print ( drop_cols )
        inp_data = self.drop_columns(inp_data, drop_cols) 
        
        return inp_data, drop_cols


data, cols = filter_df_by_corr( train_df.astype(float) )

df = data.drop( columns=cols )


print ( df.values.shape )


