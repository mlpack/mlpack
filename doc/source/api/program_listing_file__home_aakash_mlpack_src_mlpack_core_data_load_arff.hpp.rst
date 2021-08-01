
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_load_arff.hpp:

Program Listing for File load_arff.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_load_arff.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/load_arff.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_LOAD_ARFF_HPP
   #define MLPACK_CORE_DATA_LOAD_ARFF_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "dataset_mapper.hpp"
   #include <boost/tokenizer.hpp>
   
   namespace mlpack {
   namespace data {
   
   template<typename eT>
   void LoadARFF(const std::string& filename, arma::Mat<eT>& matrix);
   
   template<typename eT, typename PolicyType>
   void LoadARFF(const std::string& filename,
                 arma::Mat<eT>& matrix,
                 DatasetMapper<PolicyType>& info);
   
   } // namespace data
   } // namespace mlpack
   
   // Include implementation.
   #include "load_arff_impl.hpp"
   
   #endif
