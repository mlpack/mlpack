
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_hilbert_r_tree_descent_heuristic.hpp:

Program Listing for File hilbert_r_tree_descent_heuristic.hpp
=============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_hilbert_r_tree_descent_heuristic.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/hilbert_r_tree_descent_heuristic.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HR_TREE_DESCENT_HEURISTIC_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_HR_TREE_DESCENT_HEURISTIC_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree {
   
   class HilbertRTreeDescentHeuristic
   {
    public:
     template<typename TreeType>
     static size_t ChooseDescentNode(const TreeType* node, const size_t point);
   
     template<typename TreeType>
     static size_t ChooseDescentNode(const TreeType* node,
                                     const TreeType* insertedNode);
   };
   
   } //  namespace tree
   } //  namespace mlpack
   
   #include "hilbert_r_tree_descent_heuristic_impl.hpp"
   
   #endif // MLPACK_CORE_TREE_RECTANGLE_TREE_HR_TREE_DESCENT_HEURISTIC_HPP
