==============
u div e mean
==============

.. py:function:: u_div_e_mean(rowptr, colind, weight, feat)

   Generalized SpMM function. It fuses two steps into one kernel.

      1.Computes messages by div source node and edge features.

      2.Aggregate the messages by mean as the features on destination nodes.

   :param tensor(int) rowptr: The row pointer tensor of source feature (a sparse matrix) in CSR format.
   :param tensor(int) colind: The column index tensor of source feature (a sparse matrix) in CSR format.
   :param tensor(float) weight: The data tensor of source feature (a sparse matrix) in CSR format. 
   :param tensor(float) feat: The edge feature
   :return: The result.
   :rtype: tensor(float)