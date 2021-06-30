==========
SDDMM
==========

Sampled dense dense matrix multiplication

.. math::

    out = (D_1 \cdot D_2)\odot S

In the above equation, :math:`\odot` is a hadamard product. The sparse matrix is performed as a mask for computing dense matrix multiplication.

.. math::

out[i][j]=(\Sigma_{k} D_1[i][k]\cdot D_2[j][k])\cdot S[i][j]

We provide SDDMM in both coo and csr format.
Note that D2 is a transposed matrix.

**COO**

.. code-block:: c++

    void sddmm_cuda_coo(
            int k,          //cols for D1
            int nnz,        //non zeros number for graph
            int *rowind,    //row indices of graph
            int *colind,    //col indices of graph
            float *D1,      //left hand Dense matrix
            float *D2,      //right hand Dense matrix
            float *out);

**CSR**

.. code-block:: c++

    void sddmm_cuda_csr(
            int m,          //m for S's rows
            int k,          //k for D1's rows
            int nnz,        //non zeros number for graph
            int *rowptr,    //row pointer in CSR format
            int *colind,    //col indices of graph
            float *D1,      //left hand Dense matrix
            float *D2,      //right hand Dense matrix
            float *out);    
