=============
EdgeConv
=============

.. py:function:: class dgNN.layers.EdgeConv(self, in_feats, out_feats, batch_norm=False)

    EdgeConv layer from `Dynamic Graph CNN for Learning on Point Clouds <https://arxiv.org/pdf/1801.07829>`_

    .. math::

        h_{i}^{(l+1)}=\max _{j \in \mathcal{N}(i)}\left(\Theta \cdot\left(h_{j}^{(l)}-h_{i}^{(l)}\right)+\Phi \cdot h_{i}^{(l)}\right)

    :param in_feats: input feature size.
    :type in_feats: int
    :param out_feats: output feature size.
    :type out_feats: int
    :param batch_norm: whether to use batch normalization
    :type batch_norm: bool

    .. py:function:: forward(self, k, src_ind, feat):

        :param k: number of edges per vertex.
        :type k: int
        :param src_ind: source index tensor of shape :math:`(E)`, where :math:`E` is the number of edges.
        :type src_ind: torch.tensor
        :param feat: the input feature of shape :math:`(N,F_{in})`, where :math:`F_{in}` is the input feature size.
        :type feat: torch.tensor

        :return: the output feature of shape :math:`(N, F_{out})`, where :math:`F_{out}` is the output feature size.
        :rtype: torch.tensor





