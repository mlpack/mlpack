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
 * This class implements a way to replace target values. It is dependent on the
 * user defined StrategyType and MapperType used to hold dataset's information.
 *
 * @tparam Option of imputation strategy.
 * @tparam MapperType that is used to hold dataset information.
 * @tparam primitive type of input and output's armadillo matrix.
 */
template<typename T, typename MapperType, typename StrategyType>
class Imputer
{
 public:
  Imputer(MapperType mapper, bool transpose = true):
    mapper(std::move(mapper)),
    transpose(transpose)
  {
  // nothing to initialize here
  }

  Imputer(MapperType mapper, StrategyType strategy, bool transpose = true):
    strategy(std::move(strategy)),
    mapper(std::move(mapper)),
    transpose(transpose)
  {
  // nothing to initialize here
  }

  /**
  * Given an input dataset, replace missing values with given imputation
  * strategy.
  *
  * @param input Input dataset to apply imputation.
  * @param output
  * @oaran targetValue
  * @param mapper DatasetInfo object that holds informations about the dataset.
  * @param dimension.
  * @param transpose.
  */
  void Impute(const arma::Mat<T>& input,
              arma::Mat<T>& output,
              const std::string& missingValue,
              const size_t dimension)
  {
    T mappedValue = static_cast<T>(mapper.UnmapValue(missingValue, dimension));
    strategy.Apply(input, output, mappedValue, dimension, transpose);
  }

  /**
  * This overload of Impute() lets users to define custom value that
  * can be replaced with the target value.
  */
  void Impute(const arma::Mat<T>& input,
              arma::Mat<T>& output,
              const std::string& missingValue,
              const T& customValue,
              const size_t dimension)
  {
    T mappedValue = static_cast<T>(mapper.UnmapValue(missingValue, dimension));
    strategy.Apply(input, output, mappedValue, customValue, dimension, transpose);
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
