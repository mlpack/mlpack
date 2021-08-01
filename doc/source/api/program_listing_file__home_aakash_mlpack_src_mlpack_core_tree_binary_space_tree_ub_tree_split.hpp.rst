
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_ub_tree_split.hpp:

Program Listing for File ub_tree_split.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_ub_tree_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/binary_space_tree/ub_tree_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_UB_TREE_SPLIT_HPP
   #define MLPACK_CORE_TREE_BINARY_SPACE_TREE_UB_TREE_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "../address.hpp"
   
   namespace mlpack {
   namespace tree  {
   
   template<typename BoundType, typename MatType = arma::mat>
   class UBTreeSplit
   {
    public:
     typedef typename std::conditional<
         sizeof(typename MatType::elem_type) * CHAR_BIT <= 32,
         uint32_t,
         uint64_t>::type AddressElemType;
   
     struct SplitInfo
     {
       std::vector<std::pair<arma::Col<AddressElemType>, size_t>>* addresses;
     };
   
     bool SplitNode(BoundType& bound,
                    MatType& data,
                    const size_t begin,
                    const size_t count,
                    SplitInfo&  splitInfo);
   
     static size_t PerformSplit(MatType& data,
                                const size_t begin,
                                const size_t count,
                                const SplitInfo& splitInfo);
   
     static size_t PerformSplit(MatType& data,
                                const size_t begin,
                                const size_t count,
                                const SplitInfo& splitInfo,
                                std::vector<size_t>& oldFromNew);
   
    private:
     std::vector<std::pair<arma::Col<AddressElemType>, size_t>> addresses;
   
     void InitializeAddresses(const MatType& data);
   
     static bool ComparePair(
         const std::pair<arma::Col<AddressElemType>, size_t>& p1,
         const std::pair<arma::Col<AddressElemType>, size_t>& p2)
     {
       return bound::addr::CompareAddresses(p1.first, p2.first) < 0;
     }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "ub_tree_split_impl.hpp"
   
   #endif
