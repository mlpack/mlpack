
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_binary_numeric_split_info.hpp:

Program Listing for File binary_numeric_split_info.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_binary_numeric_split_info.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hoeffding_trees/binary_numeric_split_info.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HOEFFDING_TREES_BINARY_NUMERIC_SPLIT_INFO_HPP
   #define MLPACK_METHODS_HOEFFDING_TREES_BINARY_NUMERIC_SPLIT_INFO_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree {
   
   template<typename ObservationType = double>
   class BinaryNumericSplitInfo
   {
    public:
     BinaryNumericSplitInfo() { /* Nothing to do. */ }
     BinaryNumericSplitInfo(const ObservationType& splitPoint) :
         splitPoint(splitPoint) { /* Nothing to do. */ }
   
     template<typename eT>
     size_t CalculateDirection(const eT& value) const
     {
       return (value < splitPoint) ? 0 : 1;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(splitPoint));
     }
   
    private:
     ObservationType splitPoint;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
