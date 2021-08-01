
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_x_tree_auxiliary_information.hpp:

Program Listing for File x_tree_auxiliary_information.hpp
=========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_x_tree_auxiliary_information.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/x_tree_auxiliary_information.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_AUXILIARY_INFORMATION_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_AUXILIARY_INFORMATION_HPP
   
   namespace mlpack {
   namespace tree {
   
   template<typename TreeType>
   class XTreeAuxiliaryInformation
   {
    public:
     XTreeAuxiliaryInformation() :
       normalNodeMaxNumChildren(0),
       splitHistory(0)
     { };
   
     XTreeAuxiliaryInformation(const TreeType* node) :
         normalNodeMaxNumChildren(node->Parent() ?
             node->Parent()->AuxiliaryInfo().NormalNodeMaxNumChildren() :
             node->MaxNumChildren()),
         splitHistory(node->Bound().Dim())
     { };
   
     XTreeAuxiliaryInformation(const XTreeAuxiliaryInformation& other,
                               TreeType* /* tree */ = NULL,
                               bool /* deepCopy */ = true) :
         normalNodeMaxNumChildren(other.NormalNodeMaxNumChildren()),
         splitHistory(other.SplitHistory())
     { };
   
     XTreeAuxiliaryInformation& operator=(const XTreeAuxiliaryInformation& other)
     {
       normalNodeMaxNumChildren = other.NormalNodeMaxNumChildren();
       splitHistory = other.SplitHistory();
   
       return *this;
     }
   
     XTreeAuxiliaryInformation(XTreeAuxiliaryInformation&& other) :
         normalNodeMaxNumChildren(other.NormalNodeMaxNumChildren()),
         splitHistory(std::move(other.splitHistory))
     {
       other.normalNodeMaxNumChildren = 0;
     };
   
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
   
     bool HandlePointDeletion(TreeType* /* node */ , const size_t /* localIndex */)
     {
       return false;
     }
   
     bool HandleNodeRemoval(TreeType* /* node */ , const size_t /* nodeIndex */)
     {
       return false;
     }
   
     bool UpdateAuxiliaryInfo(TreeType* /* node */)
     {
       return false;
     }
   
     void NullifyData()
     { }
   
     typedef struct SplitHistoryStruct
     {
       int lastDimension;
       std::vector<bool> history;
   
       SplitHistoryStruct(int dim) : lastDimension(0), history(dim)
       {
         for (int i = 0; i < dim; ++i)
           history[i] = false;
       }
   
       SplitHistoryStruct(const SplitHistoryStruct& other) :
           lastDimension(other.lastDimension),
           history(other.history)
       { }
   
       SplitHistoryStruct& operator=(const SplitHistoryStruct& other)
       {
         lastDimension = other.lastDimension;
         history = other.history;
         return *this;
       }
   
       SplitHistoryStruct(SplitHistoryStruct&& other) :
           lastDimension(other.lastDimension),
           history(std::move(other.history))
       {
         other.lastDimension = 0;
       }
   
       template<typename Archive>
       void serialize(Archive& ar, const uint32_t /* version */)
       {
         ar(CEREAL_NVP(lastDimension));
         ar(CEREAL_NVP(history));
       }
     } SplitHistoryStruct;
   
    private:
     size_t normalNodeMaxNumChildren;
     SplitHistoryStruct splitHistory;
   
    public:
     size_t NormalNodeMaxNumChildren() const { return normalNodeMaxNumChildren; }
     size_t& NormalNodeMaxNumChildren() { return normalNodeMaxNumChildren; }
     const SplitHistoryStruct& SplitHistory() const { return splitHistory; }
     SplitHistoryStruct& SplitHistory() { return splitHistory; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(normalNodeMaxNumChildren));
       ar(CEREAL_NVP(splitHistory));
     }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_AUXILIARY_INFORMATION_HPP
