=================
GATConv
=================

.. code:: python

   class dgNN.layers.GATConv(self,
                            in_feats,
                            out_feats,
                            num_heads,
                            feat_drop=0.,
                            attn_drop=0.,
                            negative_slope=0.2,
                            residual=False,
                            activation=None,
                            bias=True)

* **Parameters**:
    - in_feats: input feature size
    - out_feats: output feature size
    - num_heads: number of heads in multi-head attention
    - feat_drop: dropout rate on node features
    - attn_drop: dropout rate on attention weights
    - negative_slope: negative slope for LeakyReLU
    - residual: whether to use residual connection
    - activation: applies an activation function to the updated node features
    - bias: whether to learn a bias term


Graph attention layer from `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`_

.. math:: 
    
    h_{i}^{(l+1)}=\sum_{j \in \mathcal{N}(i)} \alpha_{i, j} W^{(l)} h_{j}^{(l)}

