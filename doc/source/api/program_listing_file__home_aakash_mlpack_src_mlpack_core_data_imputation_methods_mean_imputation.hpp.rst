
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_imputation_methods_mean_imputation.hpp:

Program Listing for File mean_imputation.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_imputation_methods_mean_imputation.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/imputation_methods/mean_imputation.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEAN_IMPUTATION_HPP
   #define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEAN_IMPUTATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   template <typename T>
   class MeanImputation
   {
    public:
     void Impute(arma::Mat<T>& input,
                 const T& mappedValue,
                 const size_t dimension,
                 const bool columnMajor = true)
     {
       double sum = 0;
       size_t elems = 0; // excluding nan or missing target
   
       using PairType = std::pair<size_t, size_t>;
       // dimensions and indexes are saved as pairs inside this vector.
       std::vector<PairType> targets;
   
   
       // calculate number of elements and sum of them excluding mapped value or
       // nan. while doing that, remember where mappedValue or NaN exists.
       if (columnMajor)
       {
         for (size_t i = 0; i < input.n_cols; ++i)
         {
           if (input(dimension, i) == mappedValue ||
               std::isnan(input(dimension, i)))
           {
             targets.emplace_back(dimension, i);
           }
           else
           {
             elems++;
             sum += input(dimension, i);
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
             targets.emplace_back(i, dimension);
           }
           else
           {
             elems++;
             sum += input(i, dimension);
           }
         }
       }
   
       if (elems == 0)
         Log::Fatal << "it is impossible to calculate mean; no valid elements in "
             << "the dimension" << std::endl;
   
       // calculate mean;
       const double mean = sum / elems;
   
       // Now replace the calculated mean to the missing variables
       // It only needs to loop through targets vector, not the whole matrix.
       for (const PairType& target : targets)
       {
         input(target.first, target.second) = mean;
       }
     }
   }; // class MeanImputation
   
   } // namespace data
   } // namespace mlpack
   
   #endif
