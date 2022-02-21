==================
copy_u_sum
==================

.. code-block:: python
        
    copy_u_sum(rowptr, colind, node_feat)

Generalized SpMM function. It copies node feature to edge as the message. Then aggregates the message by **sum** on destination nodes.

**Parameters:** 
                - **rowptr** -row pointer in CSR format
                - **colind** -col indices of graph
                - **node_feat** -the source node features

**Returns:**      The result tensor

**Return type:**  tensor
