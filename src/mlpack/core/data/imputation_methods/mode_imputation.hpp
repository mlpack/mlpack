/*
 * Definition and Implementation of ModeImputation class
*/

#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MODE_IMPUTATION_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MODE_IMPUTATION_HPP

#include <mlpack/prereqs.hpp>
#include <unordered_map>

namespace mlpack {
namespace data {

/**
 * This class implements mode imputation, replacing missing values with the 
 * most frequent (mode) value in the given dimension.
 *
 * @tparam T Type of Armadillo matrix.
 */
template <typename T>
class ModeImputation
{
 public:
  /**
   * Impute function searches for mappedValue and replaces it with the 
   * mode of the given dimension. The result is overwritten in the input matrix.
   *
   * @param input Matrix that contains mappedValue.
   * @param mappedValue Value that should be replaced.
   * @param dimension Index of the dimension containing mappedValue.
   * @param columnMajor Whether the input matrix is column-major or row-major.
   */
  void Impute(arma::Mat<T>& input,
              const T& mappedValue,
              const size_t dimension,
              const bool columnMajor = true)
  {
    std::unordered_map<T, size_t> frequencyMap;
    
    if (columnMajor)
    {
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        T value = input(dimension, i);
        if (value != mappedValue && !std::isnan(value))
          frequencyMap[value]++;
      }
    }
    else
    {
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        T value = input(i, dimension);
        if (value != mappedValue && !std::isnan(value))
          frequencyMap[value]++;
      }
    }

    // Determine the mode (most frequent value)
    T mode = 0.0; // Default to 0 if all values are missing.
    size_t maxFrequency = 0;

    for (const auto& it : frequencyMap)
    {
      if (it.second > maxFrequency)
      {
        maxFrequency = it.second;
        mode = it.first;
      }
    }

    // Replace missing values with the mode
    if (columnMajor)
    {
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        if (input(dimension, i) == mappedValue || std::isnan(input(dimension, i)))
          input(dimension, i) = mode;
      }
    }
    else
    {
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        if (input(i, dimension) == mappedValue || std::isnan(input(i, dimension)))
          input(i, dimension) = mode;
      }
    }
  }
};

} // namespace data
} // namespace mlpack

#endif
