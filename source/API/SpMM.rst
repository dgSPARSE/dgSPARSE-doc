=============
SpMM
=============

Sparse matrix dense matrix multiplication

.. math::

    C = A \cdot B^T

In the above equation, A is a sparse matrix while B is a dense matrix. The outcome C is also a dense matrix. 
We provide four algorithms to choose due to their input matices. 
Plus, you could switch the layout of dense matrix into row major or column major.

.. code-block:: c++

    enum SPMV_SPMM_ALG {
        ALG_CSR_SCALAR,
        ALG_CSR_VECTOR,
        ALG_COO_SCALAR,
        ALG_COO_VECTOR
    };
    
    enum DenseLayout {
        DENSE_ROW_MAJOR,
        DENSE_COL_MAJOR
    };


The user could choose whether to compute in CSR or COO format. 
Also, you could choose the scalar method which means computing the spmm in sequential reduction on GPU.
Vector method means you could compute the spmm in parallel reduction on GPU.

.. code-block:: c++

    void cuda_csr_spmm(
            int algo_code,      //choose which algorithm
            int layout_code,    //choose which layout
            int nr,             //number of rows in A
            int nc,             //number of columns in A
            int nv,             //number of columns in B
            int nnz,            //number of nonzeros in A
            int *_csrRowPtr,    //row pointer of A
            int *_csrCol,       //column indices of A
            float *_csrVal,     //values of A
            float *_vin,        //input dense matrix (A)
            float *_vout);      //output dense matrix (C)