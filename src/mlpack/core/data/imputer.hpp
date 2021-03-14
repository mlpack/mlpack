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
#include "dataset_mapper.hpp"
#include "map_policies/missing_policy.hpp"
#include "map_policies/increment_policy.hpp"

namespace mlpack {
namespace data {

/**
 * Given a dataset of a particular datatype, replace user-specified missing
 * value with a variable dependent on the StrategyType and MapperType.
 *
 * @tparam T Type of armadillo matrix used for imputation strategy.
 * @tparam MapperType DatasetMapper that is used to hold dataset information.
 * @tparam StrategyType Imputation strategy used.
 */
template<typename T, typename MapperType, typename StrategyType>
class Imputer
{
 public:
  Imputer(MapperType mapper, bool columnMajor = true):
      mapper(std::move(mapper)),
      columnMajor(columnMajor)
  {
    // Nothing to initialize here.
  }

  Imputer(MapperType mapper, StrategyType strategy, bool columnMajor = true):
      strategy(std::move(strategy)),
      mapper(std::move(mapper)),
      columnMajor(columnMajor)
  {
    // Nothing to initialize here.
  }

  /**
  * Given an input dataset, replace missing values of a dimension with given
  * imputation strategy. This function does not produce output matrix, but
  * overwrites the result into the input matrix.
  *
  * @param input Input dataset to apply imputation.
  * @param missingValue User defined missing value; it can be anything.
  * @param dimension Dimension to apply the imputation.
  */
  void Impute(arma::Mat<T>& input,
              const std::string& missingValue,
              const size_t dimension)
  {
    T mappedValue = static_cast<T>(mapper.UnmapValue(missingValue, dimension));
    strategy.Impute(input, mappedValue, dimension, columnMajor);
  }

  //! Get the strategy.
  const StrategyType& Strategy() const { return strategy; }

  //! Modify the given strategy.
  StrategyType& Strategy() { return strategy; }

  //! Get the mapper.
  const MapperType& Mapper() const { return mapper; }

  //! Modify the given mapper.
  MapperType& Mapper() { return mapper; }

 private:
  // StrategyType
  StrategyType strategy;

  // DatasetMapperType<MapPolicy>
  MapperType mapper;

  // save columnMajor as a member variable since it is rarely changed.
  bool columnMajor;
}; // class Imputer

} // namespace data
} // namespace mlpack

#endif
