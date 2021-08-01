
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_numeric_split_info.hpp:

Program Listing for File numeric_split_info.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_numeric_split_info.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hoeffding_trees/numeric_split_info.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HOEFFDING_TREES_NUMERIC_SPLIT_INFO_HPP
   #define MLPACK_METHODS_HOEFFDING_TREES_NUMERIC_SPLIT_INFO_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree {
   
   template<typename ObservationType = double>
   class NumericSplitInfo
   {
    public:
     NumericSplitInfo() { /* Nothing to do. */ }
     NumericSplitInfo(const arma::Col<ObservationType>& splitPoints) :
         splitPoints(splitPoints) { /* Nothing to do. */ }
   
     template<typename eT>
     size_t CalculateDirection(const eT& value) const
     {
       // What bin does the point fall into?
       size_t bin = 0;
       while (bin < splitPoints.n_elem && value > splitPoints[bin])
         ++bin;
   
       return bin;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(splitPoints));
     }
   
    private:
     arma::Col<ObservationType> splitPoints;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
