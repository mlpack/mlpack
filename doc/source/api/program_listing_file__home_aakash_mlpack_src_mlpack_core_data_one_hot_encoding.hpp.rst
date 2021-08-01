
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_one_hot_encoding.hpp:

Program Listing for File one_hot_encoding.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_one_hot_encoding.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/one_hot_encoding.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_ONE_HOT_ENCODING_HPP
   #define MLPACK_CORE_DATA_ONE_HOT_ENCODING_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace data {
   
   template<typename RowType, typename MatType>
   void OneHotEncoding(const RowType& labelsIn,
                       MatType& output);
   
   template<typename eT>
   void OneHotEncoding(const arma::Mat<eT>& input,
                       const arma::Col<size_t>& indices,
                       arma::Mat<eT>& output);
   
   template<typename eT>
   void OneHotEncoding(const arma::Mat<eT>& input,
                       arma::Mat<eT>& output,
                       const data::DatasetInfo& datasetInfo);
   
   } // namespace data
   } // namespace mlpack
   
   // Include implementation.
   #include "one_hot_encoding_impl.hpp"
   
   #endif
