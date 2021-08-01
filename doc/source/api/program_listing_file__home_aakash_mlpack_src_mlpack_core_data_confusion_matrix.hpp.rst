
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_confusion_matrix.hpp:

Program Listing for File confusion_matrix.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_confusion_matrix.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/confusion_matrix.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_CONFUSION_MATRIX_HPP
   #define MLPACK_CORE_DATA_CONFUSION_MATRIX_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   
   template<typename eT>
   void ConfusionMatrix(const arma::Row<size_t> predictors,
                        const arma::Row<size_t> responses,
                        arma::Mat<eT>& output,
                        const size_t numClasses);
   
   } // namespace data
   } // namespace mlpack
   
   // Include implementation.
   #include "confusion_matrix_impl.hpp"
   
   #endif
