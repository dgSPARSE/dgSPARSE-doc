==============
edgesoftmax
==============

This is a edgesoftmax forward computation. Here we further support stacked sparse value with the same pattern by using 'head' parameter.

.. math::

    softmax_{ij}=\frac{exp(value_{ij})}{\Sigma_{j\in N(i)}exp(value_{ij})}


:math:`N(i)` is the set of nodes that have an edge to i. :math:`value_{ij}` means the value of edge point from node i to node j.

.. code-block:: c++
    
    void edge_softmax_cuda(
            int m,          //m for S's rows
            int head,       //stack parameter
            int *rowptr,    //row pointer in CSR format
            float *values,  //stacked weights of multi-graph
            float *softmax);
