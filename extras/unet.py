import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Concatenate, Add, Activation, Multiply


def create_attention_res_u_net_model(input_shape, seed):
    inputs = Input(shape=input_shape)
    initializer = tf.keras.initializers.HeNormal(seed=seed)
    
    def residual_conv_block(x, filters, pooling):
        if pooling:
            x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)  
        
        x1 = Activation("relu")(x)
        x1 = Conv2D(filters, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer)(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation("relu")(x1)
        x1 = Conv2D(filters, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer)(x1)
        
        x2 = Activation("linear")(x)
        x2 = Conv2D(filters, (1, 1), strides=(1, 1), padding="same", kernel_initializer=initializer)(x2)
        
        x3 = Add()([x1, x2])
        return x3
    
    def attention_gate(g, x):
        """
        g: gating signal
        x: skip connection
        """
        no_of_sigmoid1_filters = x.shape[3]
        sigmoid1 = Activation("relu")(
            Add()([Conv2D(no_of_sigmoid1_filters, (1, 1), strides=(2, 2), padding="same", 
                          kernel_initializer=initializer)(x),
                   Conv2D(no_of_sigmoid1_filters, (1, 1), strides=(1, 1), padding="same", 
                          kernel_initializer=initializer)(g)
                  ])
        )   
        sigmoid1 = BatchNormalization()(sigmoid1)
        sigmoid2 = Conv2D(1, (1, 1), activation="sigmoid", padding="same", kernel_initializer=initializer)(sigmoid1)
        sigmoid2 = BatchNormalization()(sigmoid2)
        alpha = UpSampling2D(interpolation="bilinear")(sigmoid2)
        x_hat = Multiply()([x, alpha])
        x_hat = BatchNormalization()(x_hat)
        return x_hat
    
    def expansion_block(g, x, filters):
        e_b = UpSampling2D(interpolation="bilinear")(g)
        e_b = BatchNormalization()(e_b)
        a = attention_gate(g, x)
        e_a = residual_conv_block(Concatenate()([a, e_b]), filters=filters, pooling=False)
        return e_a
        
    # Contraction
    c1 = residual_conv_block(x=inputs, filters=16, pooling=False)
    c2 = residual_conv_block(x=c1, filters=32, pooling=True)
    c3 = residual_conv_block(x=c2, filters=64, pooling=True)
    # c3 = residual_conv_block(x=c2, filters=64, pooling=True)
    
    # Bottleneck
    c4 = residual_conv_block(x=c3, filters=128, pooling=True)
    
    # Expansion
    e3 = expansion_block(c4, c3, 64)
    e2 = expansion_block(e3, c2, 32)
    e1 = expansion_block(e2, c1, 16)
    
    # Final
    outputs = Conv2D(1, (3, 3), activation="linear", padding="same", kernel_initializer=initializer)(e1)

    # Get the model
    model = Model(inputs=inputs, outputs=outputs) 
    model.summary()
    return model
