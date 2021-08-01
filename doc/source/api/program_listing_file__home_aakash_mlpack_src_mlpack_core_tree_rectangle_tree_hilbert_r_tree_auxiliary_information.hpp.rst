
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_hilbert_r_tree_auxiliary_information.hpp:

Program Listing for File hilbert_r_tree_auxiliary_information.hpp
=================================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_hilbert_r_tree_auxiliary_information.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/hilbert_r_tree_auxiliary_information.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HR_TREE_AUXILIARY_INFO_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_HR_TREE_AUXILIARY_INFO_HPP
   
   namespace mlpack {
   namespace tree {
   
   template<typename TreeType,
            template<typename> class HilbertValueType>
   class HilbertRTreeAuxiliaryInformation
   {
    public:
     typedef typename TreeType::ElemType ElemType;
     HilbertRTreeAuxiliaryInformation();
   
     HilbertRTreeAuxiliaryInformation(const TreeType* node);
   
     HilbertRTreeAuxiliaryInformation(
         const HilbertRTreeAuxiliaryInformation& other,
         TreeType* tree = NULL,
         bool deepCopy = true);
   
     HilbertRTreeAuxiliaryInformation(HilbertRTreeAuxiliaryInformation&& other);
   
     HilbertRTreeAuxiliaryInformation& operator=(
         const HilbertRTreeAuxiliaryInformation& other);
   
     bool HandlePointInsertion(TreeType* node, const size_t point);
   
     bool HandleNodeInsertion(TreeType* node,
                              TreeType* nodeToInsert,
                              bool insertionLevel);
   
     bool HandlePointDeletion(TreeType* node, const size_t localIndex);
   
     bool HandleNodeRemoval(TreeType* node, const size_t nodeIndex);
   
     bool UpdateAuxiliaryInfo(TreeType* node);
   
     void NullifyData();
   
     static const std::vector<TreeType*> Children(const TreeType* tree)
     { return tree->children; }
   
    private:
     HilbertValueType<ElemType> hilbertValue;
   
    public:
     const HilbertValueType<ElemType>& HilbertValue() const
     { return hilbertValue; }
     HilbertValueType<ElemType>& HilbertValue() { return hilbertValue; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #include "hilbert_r_tree_auxiliary_information_impl.hpp"
   
   #endif // MLPACK_CORE_TREE_RECTANGLE_TREE_HR_TREE_AUXILIARY_INFO_HPP
