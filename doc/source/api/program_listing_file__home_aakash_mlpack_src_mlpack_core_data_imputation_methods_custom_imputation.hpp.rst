
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_imputation_methods_custom_imputation.hpp:

Program Listing for File custom_imputation.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_imputation_methods_custom_imputation.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/imputation_methods/custom_imputation.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_CUSTOM_IMPUTATION_HPP
   #define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_CUSTOM_IMPUTATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   template <typename T>
   class CustomImputation
   {
    public:
     CustomImputation(T customValue):
         customValue(std::move(customValue))
     {
       // nothing to initialize here
     }
   
     void Impute(arma::Mat<T>& input,
                 const T& mappedValue,
                 const size_t dimension,
                 const bool columnMajor = true)
     {
       // replace the target value to custom value
       if (columnMajor)
       {
         for (size_t i = 0; i < input.n_cols; ++i)
         {
           if (input(dimension, i) == mappedValue ||
               std::isnan(input(dimension, i)))
           {
             input(dimension, i) = customValue;
           }
         }
       }
       else
       {
         for (size_t i = 0; i < input.n_rows; ++i)
         {
           if (input(i, dimension) == mappedValue ||
               std::isnan(input(i, dimension)))
           {
             input(i, dimension) = customValue;
           }
         }
       }
     }
   
    private:
     T customValue;
   }; // class CustomImputation
   
   } // namespace data
   } // namespace mlpack
   
   #endif
