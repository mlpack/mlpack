
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_r_star_tree_split.hpp:

Program Listing for File r_star_tree_split.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_r_star_tree_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/r_star_tree_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree  {
   
   class RStarTreeSplit
   {
    public:
     template <typename TreeType>
     static void SplitLeafNode(TreeType *tree, std::vector<bool>& relevels);
   
     template <typename TreeType>
     static bool SplitNonLeafNode(TreeType *tree, std::vector<bool>& relevels);
   
     template<typename TreeType>
     static size_t ReinsertPoints(TreeType* tree, std::vector<bool>& relevels);
   
     template<typename TreeType>
     static void PickLeafSplit(
         TreeType* tree,
         size_t& bestAxis,
         size_t& bestIndex);
   
    private:
     template <typename TreeType>
     static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);
   
     template<typename ElemType, typename TreeType>
     static bool PairComp(const std::pair<ElemType, TreeType>& p1,
                          const std::pair<ElemType, TreeType>& p2)
     {
       return p1.first < p2.first;
     }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation
   #include "r_star_tree_split_impl.hpp"
   
   #endif
