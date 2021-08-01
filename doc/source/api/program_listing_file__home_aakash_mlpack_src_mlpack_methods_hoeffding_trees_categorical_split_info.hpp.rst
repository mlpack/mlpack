
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_categorical_split_info.hpp:

Program Listing for File categorical_split_info.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_categorical_split_info.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hoeffding_trees/categorical_split_info.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HOEFFDING_TREES_CATEGORICAL_SPLIT_INFO_HPP
   #define MLPACK_METHODS_HOEFFDING_TREES_CATEGORICAL_SPLIT_INFO_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree {
   
   class CategoricalSplitInfo
   {
    public:
     CategoricalSplitInfo(const size_t /* categories */) { }
   
     template<typename eT>
     static size_t CalculateDirection(const eT& value)
     {
       // We have a child for each categorical value, and value should be in the
       // range [0, categories).
       return size_t(value);
     }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
