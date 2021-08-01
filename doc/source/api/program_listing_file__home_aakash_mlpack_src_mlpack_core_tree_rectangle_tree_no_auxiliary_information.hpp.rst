
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_no_auxiliary_information.hpp:

Program Listing for File no_auxiliary_information.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_no_auxiliary_information.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/no_auxiliary_information.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_NO_AUXILIARY_INFORMATION_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_NO_AUXILIARY_INFORMATION_HPP
   
   namespace mlpack {
   namespace tree {
   
   template<typename TreeType>
   class NoAuxiliaryInformation
   {
    public:
     NoAuxiliaryInformation() { }
     NoAuxiliaryInformation(const TreeType* /* node */) { }
     NoAuxiliaryInformation(const NoAuxiliaryInformation& /* other */,
                            TreeType* /* tree */,
                            bool /* deepCopy */ = true) { }
     NoAuxiliaryInformation(NoAuxiliaryInformation&& /* other */) { }
   
     NoAuxiliaryInformation& operator=(const NoAuxiliaryInformation& /* other */)
     {
       return *this;
     }
   
     bool HandlePointInsertion(TreeType* /* node */, const size_t /* point */)
     {
       return false;
     }
   
     bool HandleNodeInsertion(TreeType* /* node */,
                              TreeType* /* nodeToInsert */,
                              bool /* insertionLevel */)
     {
       return false;
     }
   
     bool HandlePointDeletion(TreeType* /* node */, const size_t /* localIndex */)
     {
       return false;
     }
   
     bool HandleNodeRemoval(TreeType* /* node */, const size_t /* nodeIndex */)
     {
       return false;
     }
   
     bool UpdateAuxiliaryInfo(TreeType* /* node */)
     {
       return false;
     }
   
     void SplitAuxiliaryInfo(TreeType* /* treeOne */,
                             TreeType* /* treeTwo */,
                             size_t /* axis */,
                             typename TreeType::ElemType /* cut */)
     { }
   
   
     void NullifyData()
     { }
   
     template<typename Archive>
     void serialize(Archive &, const uint32_t /* version */) { }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_NO_AUXILIARY_INFORMATION_HPP
