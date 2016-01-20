Data Structures
================

Input data
----------

Input data are represented by sparse matrix with row-major order.


Key value pairs
-----------------

Difacto uses key-value pairs as the major way for data communication. A key is
always an unsigned 64-bit integer, while value can be a single or a vector of
number. To store a list of pairs, we concatenate keys and values into vectors
separately, and then maintain an offset to store the position of the i-th value
in the value vector (the offset can be skipped if all values have the same
length.)


For example, assume we have three key-value pairs::

  {1, 3}, {3, 6}, {9, 3}

then we store them by::

  keys = [1, 3, 9]
  values = [3, 6, 3]

Consider pairs with vector value::

  {1, [3, 2, 4]}, {3, [6]}, {8, []}, {9, [3, 8]}

then we store them by::

  keys = [1, 3, 8, 9]
  values = [3, 2, 4, 6, 3, 8]
  value_offsets = [0, 3, 4, 4, 6]

.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/difacto/key_value.png
   :width: 400
