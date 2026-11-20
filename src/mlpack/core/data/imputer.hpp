/**
 * @file core/data/imputer.hpp
 * @author Keon Kim
 *
 * Defines Imputer class a utility function to replace missing variables in a
 * dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_IMPUTER_HPP
#define MLPACK_CORE_DATA_IMPUTER_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Given a dataset of a particular datatype, replace user-specified missing
 * value with a variable dependent on the StrategyType.
 *
 * @tparam StrategyType Imputation strategy used.
 */
template<typename StrategyType = MeanImputation>
class Imputer
{
 public:
  // Create an imputer, optionally specifying an instantiated imputation
  // strategy.
  Imputer(StrategyType strategy = StrategyType()) :
      strategy(std::move(strategy))
  {
    // Nothing to do.
  }

  /**
   * Given an input dataset, replace missing values of a dimension with given
   * imputation strategy. This function does not produce an output matrix, but
   * overwrites the result into the input matrix.
   *
   * @param input Input dataset to apply imputation.
   * @param missingValue User defined missing value; it can be anything.
   * @param dimension Dimension to apply the imputation.
   */
  template<typename MatType>
  void Impute(MatType& input,
              const typename MatType::elem_type& missingValue,
              const size_t dimension,
              const bool columnMajor = true,
              const std::enable_if_t<IsArma<MatType>::value>* = 0)
  {
    if (columnMajor && (dimension >= input.n_rows))
    {
      std::ostringstream oss;
      oss << "Imputer::Impute(): given dimension to impute (" << dimension
          << ") must be less than the number of rows in the matrix ("
          << input.n_rows << ")!" << std::endl;
      throw std::invalid_argument(oss.str());
    }
    else if (!columnMajor && (dimension >= input.n_cols))
    {
      std::ostringstream oss;
      oss << "Imputer::Impute(): given dimension to impute (" << dimension
          << ") must be less than the number of columns in the matrix ("
          << input.n_cols << ")!" << std::endl;
      throw std::invalid_argument(oss.str());
    }

    strategy.Impute(input, missingValue, dimension, columnMajor);
  }

  // Get the strategy.
  const StrategyType& Strategy() const { return strategy; }
  // Modify the given strategy.
  StrategyType& Strategy() { return strategy; }

 private:
  // StrategyType
  StrategyType strategy;
}; // class Imputer

} // namespace mlpack

#endif
