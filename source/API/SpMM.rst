=============
SpMM
=============

Sparse matrix dense matrix multiplication

.. math::

    out = S \cdot D^T

In the above equation, S is a sparse matrix while D is a dense matrix. The outcome is an (m,k) shaped matrix.
We provide SpMM in CSR format with two calculation methods, with and without value respectively.

**With Value**

.. code-block:: c++
        
    void spmm_cuda(
            int m,          //m for S's rows
            int k,          //k for D's rows
            int *rowptr,    //row pointer in CSR format
            int *colind,    //col indices of graph
            float *values,  //weights of graph
            float *dense,   //dense matrix
            float *out);


**Without Value**

.. code-block:: c++

    void spmm_cuda_no_edge_value(
            int m, 
            int k,
            int *rowptr,
            int *colind,
            float *dense,
            float *out);
