
.. _file__home_aakash_mlpack_src_mlpack_methods_range_search_rs_model.hpp:

File rs_model.hpp
=================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_range_search>` (``/home/aakash/mlpack/src/mlpack/methods/range_search``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/range_search/rs_model.hpp``)
---------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_range_search_rs_model.hpp.rst



Detailed Description
--------------------

Ryan Curtin
This is a model for range search. It is useful in that it provides an easy way to serialize a model, abstracts away the different types of trees, and also reflects the RangeSearch API and automatically directs to the right tree types.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``mlpack/core/tree/binary_space_tree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree.hpp`)

- ``mlpack/core/tree/cover_tree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_cover_tree.hpp`)

- ``mlpack/core/tree/octree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_octree.hpp`)

- ``mlpack/core/tree/rectangle_tree.hpp``

- ``range_search.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_range_search_range_search.hpp`)

- ``rs_model_impl.hpp``



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_tests_main_tests_range_search_utils.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__range`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1range_1_1LeafSizeRSWrapper`

- :ref:`exhale_class_classmlpack_1_1range_1_1RSModel`

- :ref:`exhale_class_classmlpack_1_1range_1_1RSWrapper`

- :ref:`exhale_class_classmlpack_1_1range_1_1RSWrapperBase`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/range_search/rs_model.hpp

