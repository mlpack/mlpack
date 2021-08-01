
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_r_tree_split.hpp:

Program Listing for File r_tree_split.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_r_tree_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/r_tree_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree  {
   
   class RTreeSplit
   {
    public:
     template<typename TreeType>
     static void SplitLeafNode(TreeType *tree, std::vector<bool>& relevels);
   
     template<typename TreeType>
     static bool SplitNonLeafNode(TreeType *tree, std::vector<bool>& relevels);
   
    private:
     template<typename TreeType>
     static void GetPointSeeds(const TreeType *tree, int& i, int& j);
   
     template<typename TreeType>
     static void GetBoundSeeds(const TreeType *tree, int& i, int& j);
   
     template<typename TreeType>
     static void AssignPointDestNode(TreeType* oldTree,
                                     TreeType* treeOne,
                                     TreeType* treeTwo,
                                     const int intI,
                                     const int intJ);
   
     template<typename TreeType>
     static void AssignNodeDestNode(TreeType* oldTree,
                                    TreeType* treeOne,
                                    TreeType* treeTwo,
                                    const int intI,
                                    const int intJ);
   
     template<typename TreeType>
     static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation
   #include "r_tree_split_impl.hpp"
   
   #endif
