/**
 * @file imputer.hpp
 * @author Keon Kim
 *
 * Defines Imputer class a utility function to replace missing variables in a
 * dataset.
 */
#ifndef MLPACK_CORE_DATA_IMPUTER_HPP
#define MLPACK_CORE_DATA_IMPUTER_HPP

#include <mlpack/core.hpp>
#include "dataset_info.hpp"

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
  Imputer(MapperType mapper, bool transpose = true):
    mapper(std::move(mapper)),
    transpose(transpose)
  {
    //static_assert(std::is_same<typename std::decay<MapperType>::type,
        //data::IncrementPolicy>::value, "The type of MapperType must be "
        //"IncrementPolicy");
  }

  Imputer(MapperType mapper, StrategyType strategy, bool transpose = true):
    strategy(std::move(strategy)),
    mapper(std::move(mapper)),
    transpose(transpose)
  {
    //static_assert(std::is_same<typename std::decay<MapperType>::type,
        //data::IncrementPolicy>::value, "The type of MapperType must be "
        //"IncrementPolicy");
  }

  /**
  * Given an input dataset, replace missing values with given imputation
  * strategy.
  *
  * @param input Input dataset to apply imputation.
  * @param output Armadillo matrix to save the results
  * @oaran missingValue User defined missing value; it can be anything.
  * @param dimension Dimension to apply the imputation.
  */
  void Impute(const arma::Mat<T>& input,
              arma::Mat<T>& output,
              const std::string& missingValue,
              const size_t dimension)
  {
    T mappedValue = static_cast<T>(mapper.UnmapValue(missingValue, dimension));
    strategy.Impute(input, output, mappedValue, dimension, transpose);
  }

  /**
  * This overload of Impute() lets users to define custom value that can be
  * replaced with the target value.
  */
  void Impute(const arma::Mat<T>& input,
              arma::Mat<T>& output,
              const std::string& missingValue,
              const T& customValue,
              const size_t dimension)
  {
    T mappedValue = static_cast<T>(mapper.UnmapValue(missingValue, dimension));
    strategy.Impute(input, output, mappedValue, customValue, dimension,
                    transpose);
  }

  //! Get the strategy
  const StrategyType& Strategy() const { return strategy; }

  //! Modify the given given strategy (be careful!)
  StrategyType& Strategy() { return strategy; }

  //! Get the mapper
  const MapperType& Mapper() const { return mapper; }

  //! Modify the given mapper (be careful!)
  MapperType& Mapper() { return mapper; }

 private:
  // StrategyType
  StrategyType strategy;

  // DatasetMapperType<MapPolicy>
  MapperType mapper;

  // save transpose as a member variable since it is rarely changed.
  bool transpose;

}; // class Imputer

} // namespace data
} // namespace mlpack

#endif
