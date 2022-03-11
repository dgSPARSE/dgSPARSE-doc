=====================================
GMMConv
=====================================

.. py:function:: class dgNN.layers.GMMConv(self, in_feats， out_feats, dim, n_kernels, aggregator_type='sum', residual=False, bias=True)

    Gaussian Mixture Model Convolution layer from `Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs <https://arxiv.org/abs/1611.08402>`_

    .. math::

        \begin{aligned}
        u_{i j} &=f\left(x_{i}, x_{j}\right), x_{j} \in \mathcal{N}(i) \\
        w_{k}(u) &=\exp \left(-\frac{1}{2}\left(u-\mu_{k}\right)^{T} \Sigma_{k}^{-1}\left(u-\mu_{k}\right)\right) \\
        h_{i}^{l+1} &=\text { aggregate }\left(\left\{\frac{1}{K} \sum_{k}^{K} w_{k}\left(u_{i j}\right), \forall j \in \mathcal{N}(i)\right\}\right)
        \end{aligned}


    :param in_feats: input feature size.
    :type in_feats: int
    :param out_feats: output feature size.
    :type out_feats: int
    :param dim: dimensionality of pseudo-coordinte.
    :type dim: int
    :param n_kernels: number of kernels.
    :type n_kernels: int
    :param residual: whether to use residual connection.
    :type residual: bool
    :param bias: whether to learn a bias term.
    :type bias: bool

    .. py:function:: forward(self, rowptr, colind, colptr, rowind, permute, feat, pseudo)

        :param rowptr: CSR format index pointer tensor of shape :math:`(V+1)`, where :math:`V` is the number of vertices.
        :type rowptr: torch.tensor
        :param colind: CSR format index tensor of shape :math:`(E)`, where :math:`E` is the number of edges.
        :type colind: torch.tensor
        :param colptr: CSC format index pointer tensor of shape :math:`(N+1)`.
        :type colptr: torch.tensor
        :param rowind: CSC format index tensor of shape :math:`(E)`.
        :type rowind: torch.tensor
        :param permute: csr.data[k]=csc.data[permute[k]],index tensor of shape :math:`(E)`.
        :type permute: torch.tensor
        :param feat: the input feature of shape :math:`(N,F_{in})`, where :math:`F_{in}` is the input feature size.
        :type feat: torch.tensor
        :param pseudo: pseudo coordinate tensor of shape :math:`(E, D_u)` where :math:`D_u` is the dimensionality of pseudo coordinate.
        :type pseudo: torch.tensor
        :return: the output feature of shape :math:`(N, F_{out})`, where :math:`F_{out}` is the output feature size.
        :rtype: torch.tensor





