=============
EdgeConv
=============

.. code:: python

   class dgNN.layers.EdgeConv(self,
                            in_feats,
                            out_feats,
                            batch_norm=False)
                            
* **Parameters**:
    * in_feats: input feature size
    * out_feats: output feature size
    * batch_norm: whether to use batch normalization


EdgeConv layer from `Dynamic Graph CNN for Learning on Point Clouds <https://arxiv.org/pdf/1801.07829>`_

.. math::

    h_{i}^{(l+1)}=\max _{j \in \mathcal{N}(i)}\left(\Theta \cdot\left(h_{j}^{(l)}-h_{i}^{(l)}\right)+\Phi \cdot h_{i}^{(l)}\right)

