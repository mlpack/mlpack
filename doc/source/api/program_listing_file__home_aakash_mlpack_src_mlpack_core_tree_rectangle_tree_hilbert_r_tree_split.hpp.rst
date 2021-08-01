
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_hilbert_r_tree_split.hpp:

Program Listing for File hilbert_r_tree_split.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_hilbert_r_tree_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/hilbert_r_tree_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree  {
   
   template<size_t splitOrder = 2>
   class HilbertRTreeSplit
   {
    public:
     template<typename TreeType>
     static void SplitLeafNode(TreeType* tree, std::vector<bool>& relevels);
   
     template<typename TreeType>
     static bool SplitNonLeafNode(TreeType* tree, std::vector<bool>& relevels);
   
    private:
     template<typename TreeType>
     static bool FindCooperatingSiblings(TreeType* parent,
                                         const size_t iTree,
                                         size_t& firstSibling,
                                         size_t& lastSibling);
   
     template<typename TreeType>
     static void RedistributeNodesEvenly(const TreeType* parent,
                                         const size_t firstSibling,
                                         const size_t lastSibling);
   
     template<typename TreeType>
     static void RedistributePointsEvenly(TreeType* parent,
                                          const size_t firstSibling,
                                          const size_t lastSibling);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "hilbert_r_tree_split_impl.hpp"
   
   #endif
