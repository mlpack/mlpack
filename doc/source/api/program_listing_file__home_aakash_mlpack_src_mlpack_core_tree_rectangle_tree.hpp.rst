
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree.hpp:

Program Listing for File rectangle_tree.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_HPP
   
   /* We include bounds.hpp since it gives us the necessary files.
    * However, we will not use the "ballbounds" option.
    */
   #include "bounds.hpp"
   #include "rectangle_tree/rectangle_tree.hpp"
   #include "rectangle_tree/single_tree_traverser.hpp"
   #include "rectangle_tree/single_tree_traverser_impl.hpp"
   #include "rectangle_tree/dual_tree_traverser.hpp"
   #include "rectangle_tree/dual_tree_traverser_impl.hpp"
   #include "rectangle_tree/r_tree_split.hpp"
   #include "rectangle_tree/r_star_tree_split.hpp"
   #include "rectangle_tree/no_auxiliary_information.hpp"
   #include "rectangle_tree/r_tree_descent_heuristic.hpp"
   #include "rectangle_tree/r_star_tree_descent_heuristic.hpp"
   #include "rectangle_tree/x_tree_split.hpp"
   #include "rectangle_tree/x_tree_auxiliary_information.hpp"
   #include "rectangle_tree/hilbert_r_tree_descent_heuristic.hpp"
   #include "rectangle_tree/hilbert_r_tree_split.hpp"
   #include "rectangle_tree/hilbert_r_tree_auxiliary_information.hpp"
   #include "rectangle_tree/discrete_hilbert_value.hpp"
   #include "rectangle_tree/r_plus_tree_descent_heuristic.hpp"
   #include "rectangle_tree/r_plus_tree_split_policy.hpp"
   #include "rectangle_tree/minimal_coverage_sweep.hpp"
   #include "rectangle_tree/minimal_splits_number_sweep.hpp"
   #include "rectangle_tree/r_plus_tree_split.hpp"
   #include "rectangle_tree/r_plus_plus_tree_auxiliary_information.hpp"
   #include "rectangle_tree/r_plus_plus_tree_descent_heuristic.hpp"
   #include "rectangle_tree/r_plus_plus_tree_split_policy.hpp"
   #include "rectangle_tree/traits.hpp"
   #include "rectangle_tree/typedef.hpp"
   
   #endif
