
.. _file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_spill_dual_tree_traverser.hpp:

File spill_dual_tree_traverser.hpp
==================================

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_core_tree_spill_tree>` (``/home/aakash/mlpack/src/mlpack/core/tree/spill_tree``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/core/tree/spill_tree/spill_dual_tree_traverser.hpp``)
--------------------------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_spill_dual_tree_traverser.hpp.rst



Detailed Description
--------------------

Ryan Curtin 
Marcos Pividori
Defines the SpillDualTreeTraverser for the SpillTree tree type. This is a nested class of SpillTree which traverses two trees in a depth-first manner with a given set of rules which indicate the branches which can be pruned and the order in which to recurse. The Defeatist template parameter determines if the traversers must do defeatist search on overlapping nodes.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``mlpack/prereqs.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_prereqs.hpp`)

- ``spill_dual_tree_traverser_impl.hpp``

- ``spill_tree.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_is_spill_tree.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree.hpp`




Namespaces
----------


- :ref:`namespace_mlpack`

- :ref:`namespace_mlpack__tree`


Classes
-------


- :ref:`exhale_class_classmlpack_1_1tree_1_1SpillTree_1_1SpillDualTreeTraverser`


Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/core/tree/spill_tree/spill_dual_tree_traverser.hpp

