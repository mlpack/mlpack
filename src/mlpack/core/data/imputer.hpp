/**
 * @file imputer.hpp
 * @author Keon Kim
 *
 * Defines Imputer(), a utility function to replace missing variables
 * in a dataset.
 */
#ifndef MLPACK_CORE_DATA_IMPUTER_HPP
#define MLPACK_CORE_DATA_IMPUTER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {

/**
 * Given an input dataset, replace missing values with .
 *
 * @param input Input dataset to apply imputation.
 * @param info DatasetInfo object that holds informations about the dataset.
 * @param string User-defined missing value
 * @param dimension.
 */

template<typename T>
void Imputer(arma::Mat<T>& input,
             DatasetInfo& info,
             const std::string& missingValue,
             const size_t dimension,
             const std::string& strategy)
{
  Log::Info << "impute using " << strategy << " strategy" << std::endl;

  size_t mappedValue = info.UnmapValue(missingValue, dimension);
  arma::mat stats;

  if (strategy == "mean")
  {
    stats = arma::mean(input); // mean of columns
  }
  else if (strategy == "median")
  {
    stats = arma::median(input);
  }

  for (size_t i = 0; i < input.n_cols; ++i)
  {
    if (input(dimension, i) == mappedValue)
    {
      // just for demo,
      input(dimension, i) = stats(0, i);
    }
  }
}

template<typename T>
void Imputer(arma::Mat<T>& input,
             DatasetInfo& info,
             const std::string& missingValue,
             const size_t dimension)
{
  std::string strategy = "mean"; // default strategy
  Imputer(input, info, missingValue, dimension, strategy);
}

} // namespace data
} // namespace mlpack

#endif
