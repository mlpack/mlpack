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
 * user defined Strategy and Mapper used to hold dataset's information.
 *
 * @tparam Option of imputation strategy.
 * @tparam Mapper that is used to hold dataset information.
 * @tparam primitive type of input and output's armadillo matrix.
 */
template<typename Strategy, typename Mapper, typename T>
class Imputer
{
 private:
  Strategy strat;
 public:
  Imputer()
  {
    // nothing to initialize
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
  void Impute(const arma::Mat<T> &input,
             arma::Mat<T> &output,
             const Mapper &mapper,
             const std::string &targetValue,
             const size_t dimension,
             const bool transpose = true)
  {
    auto mappedValue = mapper.UnmapValue(targetValue, dimension);
    Log::Info << "<<Imputer start>>" << std::endl;
    Log::Info << "<>mapped value<>: " << mappedValue << std::endl;
    if(transpose)
    {
      output.set_size(input.n_rows, input.n_cols);
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        Log::Info << "<Track> input=>  " << input(dimension, i) << "  mappedValue=> "<< mappedValue << std::endl;
        if (input(dimension, i) == mappedValue)
        {
          // users can specify the imputation strategies likes
          // mean, mode, etc using the class'es template parameter: Strategy.
          Log::Info << "<<IMPUTER TRANSPOSE>>" << std::endl;
          strat.template Impute<T>(input, output, dimension, i);
        }
        else
        {
          Log::Info << "<not equal>" << std::endl;
        }
      }
    }
    else
    {
      output.set_size(input.n_cols, input.n_rows);
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        if (input(i, dimension) == mappedValue)
        {
          Log::Info << "<<IMPUTER NON TRANSPOSE>>" << std::endl;
          strat.template Impute<T>(input, output, i, dimension);
        }
      }
    }
    Log::Info << "<imputer end>" << std::endl;
  }
}; // class Imputer

} // namespace data
} // namespace mlpack

#endif
