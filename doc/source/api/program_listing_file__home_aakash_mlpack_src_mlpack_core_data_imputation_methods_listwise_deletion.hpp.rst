
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_imputation_methods_listwise_deletion.hpp:

Program Listing for File listwise_deletion.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_imputation_methods_listwise_deletion.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/imputation_methods/listwise_deletion.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_LISTWISE_DELETION_HPP
   #define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_LISTWISE_DELETION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   template <typename T>
   class ListwiseDeletion
   {
    public:
     void Impute(arma::Mat<T>& input,
                 const T& mappedValue,
                 const size_t dimension,
                 const bool columnMajor = true)
     {
       std::vector<arma::uword> colsToKeep;
   
       if (columnMajor)
       {
         for (size_t i = 0; i < input.n_cols; ++i)
         {
            if (!(input(dimension, i) == mappedValue ||
                std::isnan(input(dimension, i))))
            {
              colsToKeep.push_back(i);
            }
         }
         input = input.cols(arma::uvec(colsToKeep));
       }
       else
       {
         for (size_t i = 0; i < input.n_rows; ++i)
         {
           if (!(input(i, dimension) == mappedValue ||
                std::isnan(input(i, dimension))))
           {
              colsToKeep.push_back(i);
           }
         }
         input = input.rows(arma::uvec(colsToKeep));
       }
     }
   }; // class ListwiseDeletion
   
   } // namespace data
   } // namespace mlpack
   
   #endif
