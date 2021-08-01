
.. _file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_ns_model.hpp:

File ns_model.hpp
=================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_neighbor_search>` (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search/ns_model.hpp``)
------------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_ns_model.hpp.rst



Detailed Description
--------------------

Ryan Curtin
This is a model for nearest or furthest neighbor search. It is useful in that it provides an easy way to serialize a model, abstracts away the different types of trees, and also (roughly) reflects the NeighborSearch API and automatically directs to the right tree type. It is meant to be used by the knn and kfn bindings.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``mlpack/core/tree/binary_space_tree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree.hpp`)

- ``mlpack/core/tree/cover_tree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_cover_tree.hpp`)

- ``mlpack/core/tree/octree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_octree.hpp`)

- ``mlpack/core/tree/rectangle_tree.hpp``

- ``mlpack/core/tree/spill_tree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree.hpp`)

- ``neighbor_search.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_neighbor_search.hpp`)

- ``ns_model_impl.hpp``






Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__neighbor`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1neighbor_1_1LeafSizeNSWrapper`

- :ref:`exhale_class_classmlpack_1_1neighbor_1_1NSModel`

- :ref:`exhale_class_classmlpack_1_1neighbor_1_1NSWrapper`

- :ref:`exhale_class_classmlpack_1_1neighbor_1_1NSWrapperBase`

- :ref:`exhale_class_classmlpack_1_1neighbor_1_1SpillNSWrapper`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/neighbor_search/ns_model.hpp

