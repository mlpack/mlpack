
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_normalize_labels.hpp:

Program Listing for File normalize_labels.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_normalize_labels.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/normalize_labels.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_NORMALIZE_LABELS_HPP
   #define MLPACK_CORE_DATA_NORMALIZE_LABELS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   
   template<typename eT, typename RowType>
   void NormalizeLabels(const RowType& labelsIn,
                        arma::Row<size_t>& labels,
                        arma::Col<eT>& mapping);
   
   template<typename eT>
   void RevertLabels(const arma::Row<size_t>& labels,
                     const arma::Col<eT>& mapping,
                     arma::Row<eT>& labelsOut);
   
   } // namespace data
   } // namespace mlpack
   
   // Include implementation.
   #include "normalize_labels_impl.hpp"
   
   #endif
