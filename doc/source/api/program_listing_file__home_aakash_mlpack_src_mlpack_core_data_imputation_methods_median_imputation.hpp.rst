
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_imputation_methods_median_imputation.hpp:

Program Listing for File median_imputation.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_imputation_methods_median_imputation.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/imputation_methods/median_imputation.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEDIAN_IMPUTATION_HPP
   #define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEDIAN_IMPUTATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   template <typename T>
   class MedianImputation
   {
    public:
     void Impute(arma::Mat<T>& input,
                 const T& mappedValue,
                 const size_t dimension,
                 const bool columnMajor = true)
     {
       using PairType = std::pair<size_t, size_t>;
       // dimensions and indexes are saved as pairs inside this vector.
       std::vector<PairType> targets;
       // good elements are kept inside this vector.
       std::vector<double> elemsToKeep;
   
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
             elemsToKeep.push_back(input(dimension, i));
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
              elemsToKeep.push_back(input(i, dimension));
           }
         }
       }
   
       // calculate median
       const double median = arma::median(arma::vec(elemsToKeep));
   
       for (const PairType& target : targets)
       {
          input(target.first, target.second) = median;
       }
     }
   }; // class MedianImputation
   
   } // namespace data
   } // namespace mlpack
   
   #endif
