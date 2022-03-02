=====================================
GMMConv
=====================================

.. code:: python 
    
    class dgNN.layers.GMMConv(self, 
                            in_feats, 
                            out_feats, 
                            dim, 
                            n_kernels, 
                            aggregator_type='sum', 
                            residual=False, 
                            bias=True)

* **Parameters**:
    * in_feats: input feature size
    * out_feats: output feature size
    * dim: dimensionality of pseudo-coordinte
    * n_kernels: number of kernels
    * aggregator_type: (sum, mean, max)
    * residual: whether to use residual connection
    * bias: whether to learn a bias term

Gaussian Mixture Model Convolution layer from `Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs <https://arxiv.org/abs/1611.08402>`_

.. math::

    \begin{aligned}
    u_{i j} &=f\left(x_{i}, x_{j}\right), x_{j} \in \mathcal{N}(i) \\
    w_{k}(u) &=\exp \left(-\frac{1}{2}\left(u-\mu_{k}\right)^{T} \Sigma_{k}^{-1}\left(u-\mu_{k}\right)\right) \\
    h_{i}^{l+1} &=\text { aggregate }\left(\left\{\frac{1}{K} \sum_{k}^{K} w_{k}\left(u_{i j}\right), \forall j \in \mathcal{N}(i)\right\}\right)
    \end{aligned}

