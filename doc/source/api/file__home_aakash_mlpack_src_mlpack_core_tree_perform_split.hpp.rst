
.. _file__home_aakash_mlpack_src_mlpack_core_tree_perform_split.hpp:

File perform_split.hpp
======================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_core_tree>` (``/home/aakash/mlpack/src/mlpack/core/tree``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/core/tree/perform_split.hpp``)
---------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_perform_split.hpp.rst



Detailed Description
--------------------

Mikhail Lozhnikov
This file contains functions that implement the default binary split behavior. The functions perform the actual splitting. This will order the dataset such that points that belong to the left subtree are on the left of the split column, and points from the right subtree are on the right side of the split column.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 





Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_midpoint_split.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_mean_split.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_vantage_point_split.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_rp_tree_max_split.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_rp_tree_mean_split.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__tree`

- :ref:`namespace_mlpack__tree__split`


Functions
---------


- :ref:`exhale_function_namespacemlpack_1_1tree_1_1split_1ae701f1590f5c0fb8ddea4af189f3ee8a`

- :ref:`exhale_function_namespacemlpack_1_1tree_1_1split_1a806f7a8af45201051f59a8db0b8b2feb`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/core/tree/perform_split.hpp

