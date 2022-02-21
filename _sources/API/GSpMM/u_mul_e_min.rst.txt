==================
u_mul_e_min
==================

.. code-block:: python
        
    u_mul_e_min(rowptr, colind, edge_feature, node_feat)

Generalized SpMM function. It fuses two steps into one kernel.

1. Computes messages by **mul** source node and edge features.

2. Aggregate the messages by **min** as the features on destination nodes.

**Parameters:** 
                - **rowptr** -row pointer in CSR format
                - **colind** -col indices of graph
                - **edge_feature** -the edge features
                - **node_feat** -the source node features

**Returns:**      The result tensor

**Return type:**  tensor
