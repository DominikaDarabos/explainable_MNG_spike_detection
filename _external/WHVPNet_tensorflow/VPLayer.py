import tensorflow as tf
from .ada_weighted_hermite_tf import *

tf.compat.v1.disable_eager_execution()

class VPLayer_tf(tf.keras.layers.Layer):
    def __init__(self, n_in, n_vp, vp_params, vp_penalty, **kwargs):
        super(VPLayer_tf, self).__init__(**kwargs)
        self.n_in = n_in
        self.n_out = n_vp
        self.penalty = vp_penalty
        self.nparams = len(vp_params)
        self.vp_weights = [
            self.add_weight(name=f'weight_{i}', shape=(1,), initializer=tf.keras.initializers.Constant(vp_params[i]), trainable=True)
            for i in range(self.nparams)
        ]

    @tf.custom_gradient
    def forward(self,x):
        phi, dphi, ind = ada_weighted_Hermite_tf(self.n_in, self.n_out, self.vp_weights)
        # phip = tf.linalg.pinv(phi) numerical error due to small function values
        phip = tf.linalg.pinv(phi,rcond=1e-15)
        coeffs = tf.linalg.matmul(phip, x,transpose_b=True)
        y_est = tf.transpose(tf.linalg.matmul(phi, coeffs))

        def backward(dy, variables):
            indrows = ind[0, 0]
            dx = dy @ phip

            wdphi_r = (x - y_est) @ dphi
            phipc = tf.linalg.matmul(phip, coeffs,transpose_a=True)


            def update_column(tensor,column_index,updates):
                update_flat = tf.reshape(updates, [-1])
                # eg.: [[0,0,1],[0,1,1],[0,2,1],[0,3,1],[1,0,1],[1,1,1],[1,2,1],[1,3,1]]
                x = tf.range(tf.shape(tensor)[0])
                y = tf.range(tf.shape(tensor)[1])
                z = tf.constant([column_index])
                X, Y, Z = tf.meshgrid(x, y, z, indexing='ij')
                result_tensor = tf.stack([X, Y, Z], axis=-1)
                indices = tf.reshape(result_tensor, [-1, 3])
                return tf.tensor_scatter_nd_update(tensor, indices, update_flat)

            def update_matrix(tensor,indrows,column_index,updates):
                update_flat = tf.reshape(updates, [-1])
                x = tf.range(tf.shape(tensor)[0],dtype=tf.int32)
                y = tf.cast(indrows, dtype=tf.int32)
                z = tf.constant([column_index], dtype=tf.int32)
                X, Y, Z = tf.meshgrid(x, y, z, indexing='ij')
                result_tensor = tf.stack([X, Y, Z], axis=-1)
                indices = tf.reshape(result_tensor, [-1, 3])
                return tf.tensor_scatter_nd_update(tensor, indices, update_flat)


            batch = tf.shape(x)[0]
            t2 = tf.zeros((batch,  tf.shape(phi)[1], self.nparams))
            jac1 = tf.zeros((batch, tf.shape(phi)[0], self.nparams))
            jac3 = tf.zeros((batch,  tf.shape(phi)[1], self.nparams))
            for j in range(self.nparams):
                rng = tf.equal(ind[1, :], j)
                indrows = tf.boolean_mask(ind[0, :], rng)
                jac1 = update_column(jac1,j,tf.boolean_mask(dphi, rng, axis=1) @ tf.gather(coeffs, indrows))
                t2 = update_matrix(t2,indrows,j,tf.boolean_mask(wdphi_r, rng,axis=1))
                jac3 = update_matrix(jac3,indrows,j,tf.linalg.matmul(phipc, tf.boolean_mask(dphi, rng,axis=1),transpose_a=True))

            jac = -phip @ jac1 + phip @ tf.linalg.matmul(phip,t2,transpose_a=True) + jac3 - phip @ (phi @ jac3)


            # Reshape 'dy' by adding an additional dimension with size 1
            dy = tf.expand_dims(dy, axis=-1)


            subtraction_result = tf.subtract(x, y_est)
            squared_result = tf.square(x)
            sum_squared = tf.reduce_sum(squared_result, axis=1, keepdims=True)
            res = tf.divide(subtraction_result, sum_squared)

            res = tf.expand_dims(res, axis=-1)

            grad_vars = []  # To store gradients of passed variables
            dp = tf.reduce_sum(tf.reduce_mean(jac * dy, axis=0), axis=0) - 2 * self.penalty * tf.reduce_sum(tf.reduce_mean(jac1 * res, axis=0), axis=0)
            grad_vars = [tf.reshape(dp[i], [1]) for i in range(self.nparams)]
            return dx, grad_vars

        return tf.transpose(coeffs), backward

    def call(self, inputs):
        return self.forward(inputs)
    
    # for successful keras model loading
    def get_config(self):
            config = super(VPLayer_tf, self).get_config()
            config.update({
                "n_in": self.n_in,
                "n_vp": self.n_out,
                "vp_params": self.vp_weights,
                "vp_penalty": self.penalty
            })
            return config

    @classmethod
    def from_config(cls, config):
        weight = config.pop("weight")
        layer = cls(**config)
        layer.weight.assign(weight)
        return layer
