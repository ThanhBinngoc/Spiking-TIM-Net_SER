"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Activation, Lambda, Layer
from tensorflow.keras.layers import Conv1D, SpatialDropout1D, add, GlobalAveragePooling1D, Concatenate, Multiply
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import sigmoid

@tf.custom_gradient
def surrogate_heaviside(v, threshold=0.5, alpha=10.0):
    # Chiều xuôi (Forward): Bắn xung 1 khi điện thế v >= ngưỡng, ngược lại là 0
    spike = tf.cast(v >= threshold, tf.float32)
    def grad(dy):
        # Chiều ngược (Backward): Fast Sigmoid Surrogate Gradient
        grad_v = dy / (1.0 + alpha * tf.abs(v - threshold)) ** 2
        return grad_v, None, None
    return spike, grad

class LIFGatingLayer(Layer):
    """
    Leaky Integrate-and-Fire (LIF) Spiking Gating Layer
    Mô phỏng nơ-ron sinh học:
      V(t) = tau * V(t-1) * (1 - S(t-1)) + I(t)
      S(t) = Heaviside(V(t) - threshold)
    """
    def __init__(self, tau=0.85, threshold=0.5, alpha=10.0, **kwargs):
        super(LIFGatingLayer, self).__init__(**kwargs)
        self.tau = tau
        self.threshold = threshold
        self.alpha = alpha

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        features = tf.shape(inputs)[2]
        # Chuyển trục thời gian lên đầu để quét qua chuỗi: (time, batch, features)
        inputs_t = tf.transpose(inputs, [1, 0, 2])

        def lif_step(prev_state, current_input):
            v_prev, s_prev = prev_state
            # Hard reset: nếu bước trước bắn xung (s=1) thì reset về 0, ngược lại giữ lại và nhân rò rỉ tau
            v = self.tau * v_prev * (1.0 - s_prev) + current_input
            s = surrogate_heaviside(v, self.threshold, self.alpha)
            return (v, s)

        init_v = tf.zeros((batch_size, features), dtype=inputs.dtype)
        init_s = tf.zeros((batch_size, features), dtype=inputs.dtype)
        _, spikes = tf.scan(lif_step, inputs_t, initializer=(init_v, init_s))
        # Chuyển lại về dạng chuẩn: (batch, time, features)
        return tf.transpose(spikes, [1, 0, 2])

    def get_config(self):
        config = super().get_config()
        config.update({'tau': self.tau, 'threshold': self.threshold, 'alpha': self.alpha})
        return config

def Temporal_Aware_Block(x, s, i, activation, nb_filters, kernel_size, dropout_rate=0, use_spike=False, tau=0.85, threshold=0.5, name=''):

    original_x = x
    #1.1
    conv_1_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal')(x)
    conv_1_1 = BatchNormalization(trainable=True,axis=-1)(conv_1_1)
    conv_1_1 =  Activation(activation)(conv_1_1)
    output_1_1 =  SpatialDropout1D(dropout_rate)(conv_1_1)
    # 2.1
    conv_2_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal')(output_1_1)
    conv_2_1 = BatchNormalization(trainable=True,axis=-1)(conv_2_1)
    conv_2_1 = Activation(activation)(conv_2_1)
    output_2_1 =  SpatialDropout1D(dropout_rate)(conv_2_1)
    
    if original_x.shape[-1] != output_2_1.shape[-1]:
        original_x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(original_x)
        
    output_2_1 = Activation('sigmoid')(output_2_1)

    if use_spike:
        # CẢI TIẾN: Biến cổng thành nơ-ron xung LIF
        spike_gate = LIFGatingLayer(tau=tau, threshold=threshold)(output_2_1)
        F_x = Multiply()([original_x, spike_gate])
    else:
        # Gốc: Nhân với số thực Sigmoid liên tục
        F_x = Multiply()([original_x, output_2_1])

    return F_x


class TIMNET:
    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation = "relu",
                 dropout_rate=0.1,
                 return_sequences=True,
                 use_spike=False,
                 tau=0.85,
                 threshold=0.5,
                 name='TIMNET'):
        self.name = name
        self.return_sequences = return_sequences
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.use_spike = use_spike
        self.tau = tau
        self.threshold = threshold

        self.supports_masking = True
        self.mask_value=0.

        if not isinstance(nb_filters, int):
            raise Exception()

    def __call__(self, inputs, mask=None):
        if self.dilations is None:
            self.dilations = 8
        forward = inputs
        backward = Lambda(lambda x: K.reverse(x, axes=1))(inputs)
        
        print("Input Shape=",inputs.shape)
        forward_convd = Conv1D(filters=self.nb_filters,kernel_size=1, dilation_rate=1, padding='causal')(forward)
        backward_convd = Conv1D(filters=self.nb_filters,kernel_size=1, dilation_rate=1, padding='causal')(backward)
        
        final_skip_connection = []
        
        skip_out_forward = forward_convd
        skip_out_backward = backward_convd
        
        for s in range(self.nb_stacks):
            for i in [2 ** i for i in range(self.dilations)]:
                skip_out_forward = Temporal_Aware_Block(skip_out_forward, s, i, self.activation,
                                                        self.nb_filters,
                                                        self.kernel_size, 
                                                        self.dropout_rate,  
                                                        use_spike=self.use_spike,
                                                        tau=self.tau,
                                                        threshold=self.threshold,
                                                        name=self.name)
                skip_out_backward = Temporal_Aware_Block(skip_out_backward, s, i, self.activation,
                                                        self.nb_filters,
                                                        self.kernel_size, 
                                                        self.dropout_rate,  
                                                        use_spike=self.use_spike,
                                                        tau=self.tau,
                                                        threshold=self.threshold,
                                                        name=self.name)
                
                temp_skip = add([skip_out_forward, skip_out_backward],name = "biadd_"+str(i))
                temp_skip=GlobalAveragePooling1D()(temp_skip)
                temp_skip = Lambda(lambda x: tf.expand_dims(x, axis=1))(temp_skip)
                final_skip_connection.append(temp_skip)

        x = Concatenate(axis=-2)(final_skip_connection)

        return x
