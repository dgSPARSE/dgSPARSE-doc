==============
u sub e sum
==============

.. py:function:: u_sub_e_sum(rowptr, colind, weight, feat)

   Generalized SpMM function. It fuses two steps into one kernel.

      1.Computes messages by sub source node and edge features.

      2.Aggregate the messages by sum as the features on destination nodes.

   :param tensor(int) rowptr: The row pointer tensor of source feature (a sparse matrix) in CSR format.
   :param tensor(int) colind: The column index tensor of source feature (a sparse matrix) in CSR format.
   :param tensor(float) weight: The data tensor of source feature (a sparse matrix) in CSR format. 
   :param tensor(float) feat: The edge feature
   :return: The result.
   :rtype: tensor(float)