==================
u_sub_e_mean
==================

.. code-block:: python
        
    u_sub_e_mean(rowptr, colind, edge_feature, node_feat)

Generalized SpMM function. It fuses two steps into one kernel.

1. Computes messages by **sub** source node and edge features.

2. Aggregate the messages by **mean** as the features on destination nodes.

**Parameters:** 
                - **rowptr** -row pointer in CSR format
                - **colind** -col indices of graph
                - **edge_feature** -the edge features
                - **node_feat** -the source node features

**Returns:**      The result tensor

**Return type:**  tensor
