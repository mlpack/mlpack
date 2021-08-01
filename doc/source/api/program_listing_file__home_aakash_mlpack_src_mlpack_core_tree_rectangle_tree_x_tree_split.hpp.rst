
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_x_tree_split.hpp:

Program Listing for File x_tree_split.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_x_tree_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/x_tree_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_SPLIT_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree  {
   
   const double MAX_OVERLAP = 0.2;
   
   class XTreeSplit
   {
    public:
     template<typename TreeType>
     static void SplitLeafNode(TreeType *tree, std::vector<bool>& relevels);
   
     template<typename TreeType>
     static bool SplitNonLeafNode(TreeType *tree, std::vector<bool>& relevels);
   
    private:
     template<typename TreeType>
     static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);
   
     template<typename ElemType, typename SecondType>
     static bool PairComp(const std::pair<ElemType, SecondType>& p1,
                          const std::pair<ElemType, SecondType>& p2)
     {
       return p1.first < p2.first;
     }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation
   #include "x_tree_split_impl.hpp"
   
   #endif
