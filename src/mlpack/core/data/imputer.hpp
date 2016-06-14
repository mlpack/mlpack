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
#include <mlpack/core/data/impute_strategies/custom_strategy.hpp>

namespace mlpack {
namespace data {

/**
 * This class implements a way to replace target values. It is dependent on the
 * user defined Strategy and Mapper used to hold dataset's information.
 *
 * @tparam Option of imputation strategy.
 * @tparam Mapper that is used to hold dataset information.
 * @tparam primitive type of input and output's armadillo matrix.
 */
template<typename MatType, typename Mapper, typename Strategy>
class Imputer
{
 public:
  Imputer(Mapper mapper, bool transpose =true):
    mapper(std::move(mapper)),
    transpose(transpose)
  {
    // nothing to initialize here
  }

  Imputer(Strategy strat, Mapper mapper, bool traspose = true):
    strat(std::move(strat)),
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
  void Impute(const MatType &input,
              MatType &output,
              const std::string &targetValue,
              const size_t dimension)
  {
    // find mapped value inside current mapper
    auto mappedValue = mapper.UnmapValue(targetValue, dimension);

    if(transpose)
    {
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        if (input(dimension, i) == mappedValue)
        {
          // users can specify the imputation strategies likes
          // mean, mode, etc using the class'es template parameter: Strategy.
          strat.template Impute<MatType>(input, output, dimension, i, transpose);
        }
      }
    }
    else
    {
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        if (input(i, dimension) == mappedValue)
        {
          strat.template Impute<MatType>(input, output, i, dimension, transpose);
        }
      }
    }
  }

  /**
  * This overload of Impute() lets users to define custom value that
  * can be replaced with the target value.
  */
  template <typename T>
  void Impute(const arma::Mat<T> &input,
              arma::Mat<T> &output,
              const std::string &targetValue,
              const T &customValue,
              const size_t dimension)
  {
    // find mapped value inside current mapper
    auto mappedValue = mapper.UnmapValue(targetValue, dimension);

    if(transpose)
    {
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        if (input(dimension, i) == mappedValue)
        {
          // replace the target value to custom value
          output(dimension, i) = customValue;
        }
      }
    }
    else
    {
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        if (input(i, dimension) == mappedValue)
        {
          output(i, dimension) = customValue;
        }
      }
    }
  }

  //! Get the strategy
  const Strategy& Strategy() const { return strat }

  //! Modify the given strategy (be careful!)
  Strategy& Strategy() { return strat }

  //! Get the mapper
  const Mapper& Mapper() const { return mapper }

  //! Modify the given mapper (be careful!)
  Mapper& Mapper() { return mapper }

 private:
  // Imputation Strategy
  Strategy strat;

  // DatasetMapper<MapPolicy>
  Mapper mapper;

  // save transpose as a member variable since it is rarely changed.
  bool transpose;

}; // class Imputer

} // namespace data
} // namespace mlpack

#endif
