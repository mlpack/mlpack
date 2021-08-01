
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_r_plus_plus_tree_auxiliary_information.hpp:

Program Listing for File r_plus_plus_tree_auxiliary_information.hpp
===================================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_r_plus_plus_tree_auxiliary_information.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/r_plus_plus_tree_auxiliary_information.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_AUXILIARY_INFO_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_AUXILIARY_INFO_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "../hrectbound.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename TreeType>
   class RPlusPlusTreeAuxiliaryInformation
   {
    public:
     typedef typename TreeType::ElemType ElemType;
     typedef bound::HRectBound<metric::EuclideanDistance, ElemType> BoundType;
   
     RPlusPlusTreeAuxiliaryInformation();
   
     RPlusPlusTreeAuxiliaryInformation(const TreeType* /* node */);
   
     RPlusPlusTreeAuxiliaryInformation(
         const RPlusPlusTreeAuxiliaryInformation& other,
         TreeType* tree,
         bool /* deepCopy */ = true);
   
     RPlusPlusTreeAuxiliaryInformation(RPlusPlusTreeAuxiliaryInformation&& other);
   
     bool HandlePointInsertion(TreeType* /* node */, const size_t /* point */);
   
     bool HandleNodeInsertion(TreeType* /* node */,
                              TreeType* /* nodeToInsert */,
                              bool /* insertionLevel */);
   
     bool HandlePointDeletion(TreeType* /* node */, const size_t /* localIndex */);
   
     bool HandleNodeRemoval(TreeType* /* node */, const size_t /* nodeIndex */);
   
   
     bool UpdateAuxiliaryInfo(TreeType* /* node */);
   
     void SplitAuxiliaryInfo(TreeType* treeOne,
                             TreeType* treeTwo,
                             const size_t axis,
                             const ElemType cut);
   
     void NullifyData();
   
     BoundType& OuterBound() { return outerBound; }
   
     const BoundType& OuterBound() const { return outerBound; }
   
    private:
     BoundType outerBound;
   
    public:
     template<typename Archive>
     void serialize(Archive &, const uint32_t /* version */);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #include "r_plus_plus_tree_auxiliary_information_impl.hpp"
   
   #endif // MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_AUXILIARY_INFO_HPP
