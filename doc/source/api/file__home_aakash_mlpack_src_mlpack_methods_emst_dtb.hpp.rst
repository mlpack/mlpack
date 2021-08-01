
.. _file__home_aakash_mlpack_src_mlpack_methods_emst_dtb.hpp:

File dtb.hpp
============

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_emst>` (``/home/aakash/mlpack/src/mlpack/methods/emst``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/emst/dtb.hpp``)
--------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_emst_dtb.hpp.rst



Detailed Description
--------------------

Bill March (march@gatech.edu)
Contains an implementation of the DualTreeBoruvka algorithm for finding a Euclidean Minimum Spanning Tree using the kd-tree data structure.
@inproceedings{
author={March,W.B.,Ram,P.,andGray,A.G.},
title={{FastEuclideanMinimumSpanningTree:Algorithm,Analysis,
Applications.}},
booktitle={Proceedingsofthe16thACMSIGKDDInternationalConference
onKnowledgeDiscoveryandDataMining}
series={KDD2010},
year={2010}
}

mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``dtb_impl.hpp``

- ``dtb_stat.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_emst_dtb_stat.hpp`)

- ``edge_pair.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_emst_edge_pair.hpp`)

- ``mlpack/core/metrics/lmetric.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_metrics_lmetric.hpp`)

- ``mlpack/core/tree/binary_space_tree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree.hpp`)

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)






Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__emst`


Classes
-------


- :ref:`exhale_struct_structmlpack_1_1emst_1_1DualTreeBoruvka_1_1SortEdgesHelper`

- :ref:`exhale_class_classmlpack_1_1emst_1_1DualTreeBoruvka`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/emst/dtb.hpp

