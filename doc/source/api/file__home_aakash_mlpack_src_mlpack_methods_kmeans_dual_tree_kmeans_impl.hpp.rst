
.. _file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans_impl.hpp:

File dual_tree_kmeans_impl.hpp
==============================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_kmeans>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/kmeans/dual_tree_kmeans_impl.hpp``)
----------------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans_impl.hpp.rst



Detailed Description
--------------------

Ryan Curtin
An implementation of a Lloyd iteration which uses dual-tree nearest neighbor search as a black box. The conditions under which this will perform best are probably limited to the case where k is close to the number of points in the dataset, and the number of iterations of the k-means algorithm will be few.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``dual_tree_kmeans.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans.hpp`)

- ``dual_tree_kmeans_rules.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans_rules.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__kmeans`


Functions
---------


- :ref:`exhale_function_namespacemlpack_1_1kmeans_1ae9592ed36573d5cf859276f9509ce96c`

- :ref:`exhale_function_namespacemlpack_1_1kmeans_1a2216129959ff91e21b2eafac09e891aa`

- :ref:`exhale_function_namespacemlpack_1_1kmeans_1a09ad3932f61ed9ddfbd4c5e4d91ca0e9`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/kmeans/dual_tree_kmeans_impl.hpp

