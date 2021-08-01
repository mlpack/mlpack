
.. _file__home_aakash_mlpack_src_mlpack_methods_kmeans_max_variance_new_cluster.hpp:

File max_variance_new_cluster.hpp
=================================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_kmeans>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/kmeans/max_variance_new_cluster.hpp``)
-------------------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_max_variance_new_cluster.hpp.rst



Detailed Description
--------------------

Ryan Curtin
An implementation of the EmptyClusterPolicy policy class for K-Means. When an empty cluster is detected, the point furthest from the centroid of the cluster with maximum variance is taken to be a new cluster.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``max_variance_new_cluster_impl.hpp``

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_methods_kmeans_kmeans.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__kmeans`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1kmeans_1_1MaxVarianceNewCluster`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/kmeans/max_variance_new_cluster.hpp

