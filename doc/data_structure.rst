Data Structures
================

Input data
----------

Input data are represented by sparse matrix with row-major order.


Key value pairs
-----------------

Difacto use key-value pairs for exchanging data. A key is always an unsigned
64-bit integer, while value can be have any type (int, float, ...) and arbitrary
length. To store a list of pairs, we concatenate keys and values into vector
separately, and maintain an offset to store the position of the i-th value in
the value vector (the offset can be skipped if all values have the same length.)


For example, assume we have three key-value pairs::

  {1, 8}, {3, 4}, {5, 2}

then store then by::

  keys = [1, 3, 5]
  values = [8, 4, 2]

Consider pairs with various value length::

  {1, [3, 2, 4]}, {3, [6]}, {8, []}, {9, [3, 8]}

then we store them by::

  keys = [1, 3, 8, 9]
  values = [3, 2, 4, 6, 3, 8]
  value_offsets = [0, 3, 4, 4, 6]

.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/difacto/key_value.png
   :width: 400
