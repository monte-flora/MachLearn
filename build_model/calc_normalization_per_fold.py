from DeepLearn import DeepLearn

model = DeepLearn( )

for fold in range(8):
    print ('Calculating the normalization dictionary for fold {}'.format(fold))
    training_file_names, _, _ = model.get_netcdf_file_names(fold=fold, fcst_time_idx=fcst_time_idx)
    model.get_image_normalization_params(training_file_names, 
                                        norm_dict_file_name= join(out_path,'cnn_norm_dict__fcst_time_idx={}_fold={}.json' )

    

