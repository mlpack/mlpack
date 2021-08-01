
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_r_plus_tree_split.hpp:

Program Listing for File r_plus_tree_split.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_r_plus_tree_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/r_plus_tree_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree  {
   
   template<typename SplitPolicyType,
            template<typename> class SweepType>
   class RPlusTreeSplit
   {
    public:
     typedef SplitPolicyType SplitPolicy;
     template<typename TreeType>
     static void SplitLeafNode(TreeType* tree, std::vector<bool>& relevels);
   
     template<typename TreeType>
     static bool SplitNonLeafNode(TreeType* tree, std::vector<bool>& relevels);
   
    private:
     template<typename TreeType>
     static void SplitLeafNodeAlongPartition(
         TreeType* tree,
         TreeType* treeOne,
         TreeType* treeTwo,
         const size_t cutAxis,
         const typename TreeType::ElemType cut);
   
     template<typename TreeType>
     static void SplitNonLeafNodeAlongPartition(
         TreeType* tree,
         TreeType* treeOne,
         TreeType* treeTwo,
         const size_t cutAxis,
         const typename TreeType::ElemType cut);
   
     template<typename TreeType>
     static void AddFakeNodes(const TreeType* tree, TreeType* emptyTree);
   
     template<typename TreeType>
     static bool PartitionNode(const TreeType* node,
                               size_t& minCutAxis,
                               typename TreeType::ElemType& minCut);
   
     template<typename TreeType>
     static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation
   #include "r_plus_tree_split_impl.hpp"
   
   #endif  // MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_HPP
