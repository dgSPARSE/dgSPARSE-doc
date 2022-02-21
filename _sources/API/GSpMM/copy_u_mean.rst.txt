==================
copy_u_mean
==================

.. code-block:: python
        
    copy_u_mean(rowptr, colind, node_feat)

Generalized SpMM function. It copies node feature to edge as the message. Then aggregates the message by **mean** on destination nodes.

**Parameters:** 
                - **rowptr** -row pointer in CSR format
                - **colind** -col indices of graph
                - **node_feat** -the source node features

**Returns:**      The result tensor

**Return type:**  tensor
