
.. _file__home_aakash_mlpack_src_mlpack_methods_rann_ra_search.hpp:

File ra_search.hpp
==================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_rann>` (``/home/aakash/mlpack/src/mlpack/methods/rann``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/rann/ra_search.hpp``)
--------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_rann_ra_search.hpp.rst



Detailed Description
--------------------

Parikshit Ram
Defines the RASearch class, which performs an abstract rank-approximate nearest/farthest neighbor query on two datasets.
The details of this method can be found in the following paper:
@inproceedings{ram2009rank,
title={{Rank-ApproximateNearestNeighborSearch:RetainingMeaningand
SpeedinHighDimensions}},
author={{Ram,P.andLee,D.andOuyang,H.andGray,A.G.}},
booktitle={{AdvancesofNeuralInformationProcessingSystems}},
year={2009}
}

mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``mlpack/core/metrics/lmetric.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_metrics_lmetric.hpp`)

- ``mlpack/core/tree/binary_space_tree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree.hpp`)

- ``mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_sort_policies_nearest_neighbor_sort.hpp`)

- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)

- ``ra_query_stat.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_rann_ra_query_stat.hpp`)

- ``ra_search_impl.hpp``

- ``ra_typedef.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_rann_ra_typedef.hpp`)

- ``ra_util.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_rann_ra_util.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_methods_rann_ra_model.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_rann_ra_typedef.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__neighbor`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1neighbor_1_1LeafSizeRAWrapper`

- :ref:`exhale_class_classmlpack_1_1neighbor_1_1RASearch`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/rann/ra_search.hpp

