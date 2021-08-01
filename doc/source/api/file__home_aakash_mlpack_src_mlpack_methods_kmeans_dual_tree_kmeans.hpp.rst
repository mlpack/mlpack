
.. _file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans.hpp:

File dual_tree_kmeans.hpp
=========================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_kmeans>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/kmeans/dual_tree_kmeans.hpp``)
-----------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans.hpp.rst



Detailed Description
--------------------

Ryan Curtin
An implementation of a Lloyd iteration which uses dual-tree nearest neighbor search as a black box. The conditions under which this will perform best are probably limited to the case where k is close to the number of points in the dataset, and the number of iterations of the k-means algorithm will be few.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``dual_tree_kmeans_impl.hpp``

- ``dual_tree_kmeans_statistic.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans_statistic.hpp`)

- ``mlpack/core/tree/binary_space_tree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree.hpp`)

- ``mlpack/core/tree/cover_tree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_cover_tree.hpp`)

- ``mlpack/methods/neighbor_search/neighbor_search.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_neighbor_search.hpp`)






Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__kmeans`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1kmeans_1_1DualTreeKMeans`


Functions
---------


- :ref:`exhale_function_namespacemlpack_1_1kmeans_1ae824715a9723b95291ede5396d7ec48a`

- :ref:`exhale_function_namespacemlpack_1_1kmeans_1ad50ab0e1083d84c7f78a79f2bdaec558`

- :ref:`exhale_function_namespacemlpack_1_1kmeans_1ab8a2dc63dd61b947e7b90e03d31d64c0`

- :ref:`exhale_function_namespacemlpack_1_1kmeans_1aef92bb2544a815fc97f3c79070e5d3c0`


Typedefs
--------


- :ref:`exhale_typedef_namespacemlpack_1_1kmeans_1a050f8eba1b8d0c72e990a9ee3b7ed775`

- :ref:`exhale_typedef_namespacemlpack_1_1kmeans_1a3d8c82eb428be782996066d70afc122b`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/kmeans/dual_tree_kmeans.hpp

