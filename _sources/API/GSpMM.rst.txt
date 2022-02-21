=============
GSpMM
=============

Generalized Sparse-Matrix Dense-Matrix Multiplication operators, which are proved to be the building blocks in GNN. 
Here we only implement the forward propagation of GSpMM. 
It functions is as follows. 
1. Computes messages by **add/sub/mul/div** source node and edge features, or copy node features to edges.

2. Aggregate the messages by **sum/max/min/mean** as the features on destination nodes.

The following are all supported operations:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    GSpMM/u_add_e_sum
    GSpMM/u_sub_e_sum
    GSpMM/u_mul_e_sum
    GSpMM/u_div_e_sum
    GSpMM/u_add_e_max
    GSpMM/u_sub_e_max
    GSpMM/u_mul_e_max
    GSpMM/u_div_e_max
    GSpMM/u_add_e_min
    GSpMM/u_sub_e_min
    GSpMM/u_mul_e_min
    GSpMM/u_div_e_min
    GSpMM/u_add_e_mean
    GSpMM/u_sub_e_mean
    GSpMM/u_mul_e_mean
    GSpMM/u_div_e_mean
    GSpMM/copy_u_sum
    GSpMM/copy_u_max
    GSpMM/copy_u_min
    GSpMM/copy_u_mean
