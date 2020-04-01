from DeepLearn import DeepLearn, find_model_metafile, read_model_metadata, read_keras_model
from DeepLearn import write_model_metadata
from build_convolution_neural_network import ConvNet
from sklearn.isotonic import IsotonicRegression
from joblib import dump, load

generic_file_name_str = '/DL_WOFS_*_{}_patches.nc'.format(6)
model_builder = ConvNet( )

params = {  'num_conv_blocks':2,
                'num_conv_layers_in_a_block': 2,
                'num_dense_layers':3,
                'first_num_filters':16,
                'use_batch_normalization':False,
                'kernel_size':5,
                'dropout_fraction': 0.5,
                'l1_weight':0.0,
                'l2_weight':0.001,
                'activation_function_name':'leaky_relu',
                'pooling_type':'mean',
                'dim_option':'2D',
                'conv_type': 'traditional',
                'min_delta': 1e-5,
                'patience': 6
                }

model = DeepLearn( ) 
training_file_names, validation_file_names, testing_file_names = model.get_netcdf_file_names(fold=0, generic_file_name_str=generic_file_name_str)
cnn_file_name = 'cnn_model.h5'
cnn_model_object = model_builder.build_model( input_shape = (24,24,33), cnn_model_params=params)

normalization_dict = model.get_image_normalization_params(training_file_names)
cnn_metadata_dict = model.train_cnn(
    cnn_model_object=cnn_model_object,
    training_file_names=training_file_names,
    num_examples_per_batch=256, num_epochs=10,
    num_training_batches_per_epoch=10,
    validation_file_names=validation_file_names,
    num_validation_batches_per_epoch=10,
    output_model_file_name=cnn_file_name,
    normalization_dict=normalization_dict)

write_model_metadata( model_metadata_dict=cnn_metadata_dict,
                      json_file_name = 'cnn_model_metadata.json'
                      )
cnn_metafile_name = find_model_metafile(cnn_file_name)
cnn_metadata_dict = read_model_metadata('cnn_model_metadata.json')
validation_image_dict = model.read_many_netcdf_files(validation_file_names)

validation_predictions  = model.evaluate_cnn(
        cnn_model_object = read_keras_model(cnn_file_name),
        calibrated_model = None, 
        image_dict = validation_image_dict,
        cnn_metadata_dict = cnn_metadata_dict
        )

print ('Training the isotonic regression model...')
calibrated_clf = IsotonicRegression( out_of_bounds='clip' )
calibrated_clf.fit(validation_predictions.astype(float), validation_image_dict['target_matrix'])

dump(calibrated_clf, 'iso_model_for_cnn.joblib')

validation_predictions  = model.evaluate_cnn(
        cnn_model_object = read_keras_model(cnn_file_name),
        calibrated_model = calibrated_clf,           
        image_dict = validation_image_dict,
        cnn_metadata_dict = cnn_metadata_dict
        )


