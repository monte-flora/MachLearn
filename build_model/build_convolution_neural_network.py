import keras
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, Conv3D,  Input, AveragePooling2D, AveragePooling3D, Flatten, LeakyReLU
from keras.layers import Dropout, BatchNormalization, ELU, MaxPooling2D, MaxPooling3D, ActivityRegularization
from keras.layers import SeparableConv2D
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import keras.backend as K
import numpy as np

def brier_score_keras(obs, preds):
    return K.mean((preds - obs) ** 2)

def brier_skill_score_keras(obs, preds):
    climo = K.mean((obs - K.mean(obs)) ** 2)
    return 1.0 - brier_score_keras(obs, preds) / climo


class ConvNet:
    """
    ConvNet builds convolution neural networks
    """
    def build_model( self, input_shape, cnn_model_params ):  
        '''
        Builds the keras model.
        Args:
        ===========
            input_shape, array, shape of the input array 
            cnn_model_params, dict, dictionary with the convolution neural network model parameters
                Keys:
                num_conv_blocks, int, number of convolution 'blocks' (composed of one or more stacks of convolution layers)
                num_dense_layers, int, number of dense layers following the convolution layers 
                num_conv_layers_in_a_block, int, number of convolution layers in a convolution 'block'
                first_num_filters, int, number of filters used in the first convolution layer
                                filter count is then doubled for each additional layer 
                use_batch_normalization, boolean, if True apply batch normalization after activation of each of the convolution layers
                kernel_size, int, kernel size for the convolution filter
                dropout_fraction, float, the dropout rate used in the training of the dense layers (bewteen 0-1)
                l1_weight, float, L1 regularization weight 
                l2_weight, float, L2 regularization weight
                activation_function_name, str, name of the activation function being used 
                    option: 
                        'leaky_relu', 
                        'elu', 
                pooling_type, str, type of pooling being used 
                    option: 
                        'max', max pooling 
                        'mean', average pooling 
                dim_option, str, dimension of the convolution layer 
                    option:
                        '2D', two-dimensional convolution layers
                        '3D', three-dimensional convolution layers 
                conv_type, str, 
                    option:
                        'traditional',
                        'separable',   
        Returns:
        ==========
            conv_model, keras model object (not fitted)  
        '''
        input_layer_object = self._get_input_layer( input_shape)
        last_layer_object = input_layer_object

        next_num_filters = None

        # Number of convolution blocks 
        for _ in range(cnn_model_params['num_conv_blocks']):
            for i in range(cnn_model_params['num_conv_layers_in_a_block']):
                if next_num_filters is None:
                    next_num_filters = cnn_model_params['first_num_filters'] + 0
                else:
                    next_num_filters = 2 * next_num_filters
                # Convolution layer 
                #print next_num_filters 
                #print _ , 'conv _layer' , next_num_filters 
                conv_layer_object = self._get_convolution_layer( num_filters = next_num_filters, kernel_size = cnn_model_params['kernel_size'], dim_option = cnn_model_params['dim_option'] )
                last_layer_object = conv_layer_object(last_layer_object)

                # Apply activation 
                activation_layer_object = self._get_activation_layer( function_name = cnn_model_params['activation_function_name'] )
                last_layer_object = activation_layer_object(last_layer_object)

                # Apply batch normalization (optional) 
                if cnn_model_params['use_batch_normalization']:
                    batch_norm_layer_object = self._get_batch_norm_layer( )
                    last_layer_object       = batch_norm_layer_object( last_layer_object )
            # Apply pooling 
            pooling_layer_object = self._get_pooling_layer( cnn_model_params['pooling_type'], cnn_model_params['dim_option'] )
            last_layer_object    = pooling_layer_object(last_layer_object)

        these_dimensions = np.array(last_layer_object.get_shape().as_list()[1:], dtype=int)
        num_scalar_features = np.prod(these_dimensions)

        # Flatten the convolution layer output 
        flattening_layer_object = Flatten()
        last_layer_object = flattening_layer_object(last_layer_object)

        dummy, num_outputs_by_dense_layer = self._get_dense_layer_dimensions(num_features=num_scalar_features, num_predictions=1, num_dense_layers= cnn_model_params['num_dense_layers'])

        # Number of dense layers 
        for i in range(  cnn_model_params['num_dense_layers'] ):
            # print i, 'dense layers' , num_outputs_by_dense_layer[i]
            # Dense layer 
            dense_layer_object = self._get_dense_layer( num_outputs_by_dense_layer[i],  cnn_model_params['l1_weight'],  cnn_model_params['l2_weight'] )
            last_layer_object  = dense_layer_object( last_layer_object )

            # The output layer 
            if i == cnn_model_params['num_dense_layers'] - 1:
                activation_layer_object = self._get_activation_layer(function_name='sigmoid')
                last_layer_object = activation_layer_object(last_layer_object)
                break

            # Apply activation layer 
            activation_layer_object = self._get_activation_layer( function_name =  cnn_model_params['activation_function_name'] )
            last_layer_object = activation_layer_object(last_layer_object)

            # Apply batch normalization (optional) 
            if  cnn_model_params['use_batch_normalization']:
                batch_norm_layer_object = self._get_batch_norm_layer()
                last_layer_object = batch_norm_layer_object(last_layer_object)

            # Apply weight dropout (optional)
            if  cnn_model_params['dropout_fraction'] > 0:
                dropout_layer_object = self._get_dropout_layer(  cnn_model_params['dropout_fraction'] )
                last_layer_object = dropout_layer_object(last_layer_object)

        conv_model = Model( inputs=input_layer_object, outputs=last_layer_object)
        conv_model.compile(optimizer = Adam( ), loss = "binary_crossentropy", metrics=['mse'])

        print((conv_model.summary()))

        return conv_model

    def _get_input_layer( self, input_shape ): 
        """ Creates the input layer.
        """
        return Input( shape = input_shape ) 

    def _get_activation_layer(self, function_name, alpha_parameter=0.2): 
        """ Creates an activation layer. 
        :param function name: Name of activation function (must be accepted by
                        `_check_activation_function`).
        :param alpha_parameter: Slope (used only for eLU and leaky ReLU functions).
        :return: layer_object: Instance of `keras.layers.Activation`,
                        `keras.layers.ELU`, or `keras.layers.LeakyReLU`.
        """
        if function_name == 'elu': 
            return ELU( alpha = alpha_parameter )
        if function_name == 'leaky_relu': 
            return LeakyReLU( alpha = alpha_parameter) 
        return Activation(function_name)  

    def _get_convolution_layer( self, num_filters, kernel_size, dim_option = '2D', conv_type = 'traditional' ): 
        """ Creates a 2D or 3D convolution layer.
        """ 
        if dim_option == '2D': 
            if conv_type == 'traditional': 
               return Conv2D( filters     = num_filters, 
                            kernel_size = (kernel_size, kernel_size),
                            padding     = 'same',
                            use_bias    = True , 
                            activation  = None , 
                            data_format = 'channels_last' )

            elif conv_type == 'separable':    
                return SeparableConv2D( filters     = num_filters, 
                            kernel_size = (kernel_size, kernel_size), 
                            padding     = 'same', 
                            use_bias    = True ,
                            activation  = None , 
                            data_format = 'channels_last' )  
        if dim_option == '3D': 
            return Conv3D(  filters     = num_filters,
                                        kernel_size = (kernel_size, kernel_size, kernel_size), 
                                        padding     = 'same',
                                        use_bias    = True ,
                                        activation  = None ,
                                        data_format = 'channels_last' )
    def _get_dense_layer( self, num_neurons, l1_weight, l2_weight ): 
        """ Create a Dense layer with optionally regularization. 
        """
        return Dense( num_neurons ,
                              kernel_initializer = 'glorot_uniform',
                              use_bias           = True,
                              bias_initializer   = 'zeros',
                              activation         = None,
                              kernel_regularizer = self._get_regularization_layer( l1_weight, l2_weight) )

    def _get_regularization_layer(self,  l1_weight, l2_weight ):
        """ Creates a regularization object.
        """
        return keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight)

    def _get_pooling_layer( self, pooling_type, dim_option = '2D' ):
        """ Creates 2-D or 3-D pooling layer.
        """
        if dim_option == '2D':
            if pooling_type == 'max':
                return MaxPooling2D( )
            if pooling_type == 'mean':
                return AveragePooling2D( )

        if dim_option == '3D':
            if  pooling_type == 'max':
                return MaxPooling3D( )
            if pooling_type == 'mean':
                return AveragePooling3D( )

    def _get_batch_norm_layer( self ):
            """Creates batch-normalization layer.

            :return: layer_object: Instance of `keras.layers.BatchNormalization`.
            """
            return BatchNormalization( axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)

    def _get_dropout_layer( self, dropout_fraction ):
        """ Create a dropout object for the dense layers
        """
        return Dropout( rate = dropout_fraction )

    def _get_dense_layer_dimensions( self, num_features, num_predictions, num_dense_layers):
            """Returns dimensions (num input and output features) for each dense layer.

            D = number of dense layers

            :param num_features: Number of features (inputs to the first dense layer).
            :param num_predictions: Number of predictions (outputs from the last dense
                    layer).
            :param num_dense_layers: Number of dense layers.
            :return: num_inputs_by_layer: length-D np array with number of input
                features per dense layer.
            :return: num_outputs_by_layer: length-D np array with number of output
                features per dense layer.
            """

            e_folding_param = (
                float(-1 * num_dense_layers) /
                np.log(float(num_predictions) / num_features) )

            dense_layer_indices = np.linspace(
                0, num_dense_layers - 1, num=num_dense_layers, dtype=float)
            num_inputs_by_layer = num_features * np.exp(
                -1 * dense_layer_indices / e_folding_param)
            num_inputs_by_layer = np.round(num_inputs_by_layer).astype(int)

            num_outputs_by_layer = np.concatenate((
                num_inputs_by_layer[1:],
                np.array([num_predictions], dtype=int)
                ))

            return num_inputs_by_layer, num_outputs_by_layer


