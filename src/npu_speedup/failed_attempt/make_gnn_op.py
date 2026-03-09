
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras import layers, Model

# def make_ops_2_to_2_npu(m=32, d=32):
#     inputs = layers.Input(shape=(m, m, d))
    
#     # 1. Identity (Op 10)
#     op10 = inputs
    
#     # 2. Transpose (Op 11) -> Use the standard Permute layer
#     # Note: Permute uses 1-based indexing for the input_shape (excluding batch)
#     # Original: (m, m, d) -> We want (width, height, channels)
#     op11 = layers.Permute((2, 1, 3), name="transpose")(inputs)
    
#     # 3. Row Sums (Op 7) -> Use DepthwiseConv2D for "Hardware-Accelerated" summing
#     # A DepthwiseConv with a kernel of ones is a very fast way to sum across an axis
#     # To sum across a row (m columns), we use a (1, m) kernel
#     row_sum_kernel = np.ones((1, m, d, 1)).astype('float32')
#     row_sums = layers.DepthwiseConv2D(
#         kernel_size=(1, m), 
#         use_bias=False, 
#         depthwise_initializer=tf.keras.initializers.Constant(row_sum_kernel),
#         trainable=False,
#         name="row_sum_dw"
#     )(inputs) # Output: (1, m, 1, d)
    
#     # Tiling (Broadcasting) is handled automatically by hardware if we use an "UpSampling" 
#     # or just a simple multiplication by ones. Let's use UpSampling2D:
#     op7 = layers.UpSampling2D(size=(1, m), name="op7_broadcast")(row_sums)
    
#     # 4. Col Sums (Op 6)
#     col_sum_kernel = np.ones((m, 1, d, 1)).astype('float32')
#     col_sums = layers.DepthwiseConv2D(
#         kernel_size=(m, 1), 
#         use_bias=False, 
#         depthwise_initializer=tf.keras.initializers.Constant(col_sum_kernel),
#         trainable=False,
#         name="col_sum_dw"
#     )(inputs) # Output: (1, 1, m, d)
    
#     op6 = layers.UpSampling2D(size=(m, 1), name="op6_broadcast")(col_sums)
    
#     # 5. Diagonal (Op 1)
#     diag_mask = np.eye(m).reshape(1, m, m, 1).astype('float32')
#     op1 = layers.Multiply(name="op1")([inputs, tf.constant(diag_mask)])
    
#     # Combine
#     output = layers.Concatenate(axis=-1)([op1, op6, op7, op10, op11])
    
#     model = Model(inputs=inputs, outputs=output)
#     model.save('gnn_ops_32.h5')
#     print("✅ Successfully built a Lambda-free model for MemryX!")

# if __name__ == "__main__":
#     make_ops_2_to_2_npu()


import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model

def make_ops_2_to_2_npu(m=32, d=32):
    inputs = layers.Input(shape=(m, m, d))
    
    # 1. Identity (Op 10)
    op10 = inputs
    
    # 2. Transpose (Op 11)
    op11 = layers.Permute((2, 1, 3), name="transpose")(inputs)
    
    # 3. Row Sums (Op 7) - (Sum across rows, then broadcast)
    row_sum_kernel = np.ones((1, m, d, 1)).astype('float32')
    row_sums = layers.DepthwiseConv2D(
        kernel_size=(1, m), 
        use_bias=False, 
        depthwise_initializer=tf.keras.initializers.Constant(row_sum_kernel),
        trainable=False,
        name="row_sum_dw"
    )(inputs) 
    op7 = layers.UpSampling2D(size=(1, m), name="op7_broadcast")(row_sums)
    
    # 4. Col Sums (Op 6) - (Sum across columns, then broadcast)
    col_sum_kernel = np.ones((m, 1, d, 1)).astype('float32')
    col_sums = layers.DepthwiseConv2D(
        kernel_size=(m, 1), 
        use_bias=False, 
        depthwise_initializer=tf.keras.initializers.Constant(col_sum_kernel),
        trainable=False,
        name="col_sum_dw"
    )(inputs)
    op6 = layers.UpSampling2D(size=(m, 1), name="op6_broadcast")(col_sums)
    
    # 5. Diagonal Extraction (Op 1)
    # Instead of Multiply(inputs, constant), we use a 1x1 DepthwiseConv
    # where the weights are 1 on the diagonal and 0 elsewhere.
    diag_mask = np.eye(m).reshape(m, m, 1, 1).astype('float32')
    # We repeat the mask for all d channels
    full_diag_mask = np.tile(diag_mask, [1, 1, d, 1])
    
    op1 = layers.DepthwiseConv2D(
        kernel_size=(m, m),
        use_bias=False,
        padding='same', # Keeps it m x m
        depthwise_initializer=tf.keras.initializers.Constant(full_diag_mask),
        trainable=False,
        name="op1_diag_conv"
    )(inputs)
    
    # Final Concatenate
    output = layers.Concatenate(axis=-1)([op1, op6, op7, op10, op11])
    
    model = Model(inputs=inputs, outputs=output)
    model.save('gnn_ops_32.h5')
    print("✅ Model updated: Multiply replaced with DepthwiseConv for better MPU compatibility.")

if __name__ == "__main__":
    make_ops_2_to_2_npu()