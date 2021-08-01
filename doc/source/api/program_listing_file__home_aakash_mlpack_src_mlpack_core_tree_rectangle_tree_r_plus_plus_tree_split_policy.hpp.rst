
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_r_plus_plus_tree_split_policy.hpp:

Program Listing for File r_plus_plus_tree_split_policy.hpp
==========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_r_plus_plus_tree_split_policy.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/r_plus_plus_tree_split_policy.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_SPLIT_POLICY_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_SPLIT_POLICY_HPP
   
   namespace mlpack {
   namespace tree {
   
   class RPlusPlusTreeSplitPolicy
   {
    public:
     static const int SplitRequired = 0;
     static const int AssignToFirstTree = 1;
     static const int AssignToSecondTree = 2;
   
     template<typename TreeType>
     static int GetSplitPolicy(const TreeType& child,
                               const size_t axis,
                               const typename TreeType::ElemType cut)
     {
       if (child.AuxiliaryInfo().OuterBound()[axis].Hi() <= cut)
         return AssignToFirstTree;
       else if (child.AuxiliaryInfo().OuterBound()[axis].Lo() >= cut)
         return AssignToSecondTree;
   
       return SplitRequired;
     }
   
     template<typename TreeType>
     static const
         bound::HRectBound<metric::EuclideanDistance, typename TreeType::ElemType>&
             Bound(const TreeType& node)
     {
       return node.AuxiliaryInfo().OuterBound();
     }
   };
   
   } //  namespace tree
   } //  namespace mlpack
   
   #endif //  MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_SPLIT_POLICY_HPP
