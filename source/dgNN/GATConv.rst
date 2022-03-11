=================
GATConv
=================

.. py:function:: class dgNN.layers.GATConv(self, in_feats, out_feats, num_heads, feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False, activation=None, bias=True)

    Graph attention layer from `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`_

    .. math:: 
    
        h_{i}^{(l+1)}=\sum_{j \in \mathcal{N}(i)} \alpha_{i, j} W^{(l)} h_{j}^{(l)}

    :param in_feats: input feature size.
    :type in_feats: int
    :param out_feats: output feature size.
    :type out_feats: int
    :param num_heads: number of heads in multi-head attention.
    :type num_heads: int
    :param feat_drop: dropout rate on node features.
    :type feat_drop: float
    :param attn_drop: dropout rate on attention weights.
    :type attn_drop: float
    :param negative_slope: negative slope for LeakyReLU.
    :type negative_slope: float
    :param residual: whether to use residual connection.
    :type residual: bool
    :param bias: whether to learn a bias term.
    :type bias: bool

    .. py:function:: forward(self, row_ptr, col_ind, col_ptr, row_ind, feat)

        :param row_ptr: CSR format index pointer tensor of shape :math:`(V+1)`, where :math:`V` is the number of vertices.
        :type row_ptr: torch.tensor
        :param col_ind: CSR format index tensor of shape :math:`(E)`, where :math:`E` is the number of edges.
        :type col_ind: torch.tensor
        :param col_ptr: CSC format index pointer tensor of shape :math:`(N+1)`.
        :type col_ptr: torch.tensor
        :param row_ind: CSC format index tensor of shape :math:`(E)`.
        :type row_ind: torch.tensor
        :param feat: the input feature of shape :math:`(N,F_{in})`, where :math:`F_{in}` is the input feature size.
        :type feat: torch.tensor

        :return: the output feature of shape :math:`(N, H, F_{out})`, where :math:`F_{out}` is the output feature size and :math:`H` is the number of attention heads.
        :rtype: torch.tensor







