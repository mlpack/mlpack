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
template<typename MatType, typename Mapper, typename Strategy = CustomStrategy>
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
  void Impute(const MatType &input,
             MatType &output,
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
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        Log::Info << "<Track> input=>  " << input(dimension, i) << "  mappedValue=> "<< mappedValue << std::endl;
        if (input(dimension, i) == mappedValue)
        {
          // users can specify the imputation strategies likes
          // mean, mode, etc using the class'es template parameter: Strategy.
          Log::Info << "<<IMPUTER TRANSPOSE>>" << std::endl;
          strat.template Impute<MatType>(input, output, dimension, i, transpose);
        }
        else
        {
          Log::Info << "<not equal>" << std::endl;
        }
      }
    }
    else
    {
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        Log::Info << "<Track> input=>  " << input(dimension, i) << "  mappedValue=> "<< mappedValue << std::endl;
        if (input(i, dimension) == mappedValue)
        {
          Log::Info << "<<IMPUTER NON TRANSPOSE>>" << std::endl;
          strat.template Impute<T>(input, output, i, dimension, transpose);
        }
        else {
          Log::Info << "<not equal>" << std::endl;
        }
      }
    }
    Log::Info << "<imputer end>" << std::endl;
  }

  /**
  * This overload of Impute() lets users to define custom value that
  * can be replaced with the target value.
  */
  template <typename T>
  void Impute(const arma::Mat<T> &input,
              arma::Mat<T> &output,
              const Mapper &mapper,
              const std::string &targetValue,
              const T &customValue,
              const size_t dimension,
              const bool transpose = true)
  {
    auto mappedValue = mapper.UnmapValue(targetValue, dimension);
    Log::Info << "<<CUSTOM Imputer start>>" << std::endl;
    Log::Info << "<>mapped value<>: " << mappedValue << std::endl;
    if(transpose)
    {
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        Log::Info << "<Track> input=>  " << input(dimension, i) << "  mappedValue=> "<< mappedValue << std::endl;
        if (input(dimension, i) == mappedValue)
        {
          // replace the target value to custom value
          Log::Info << "<<IMPUTER TRANSPOSE>>" << std::endl;
          output(dimension, i) = customValue;
        }
        else
        {
          Log::Info << "<not equal>" << std::endl;
        }
      }
    }
    else
    {
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        Log::Info << "<Track> input=>  " << input(dimension, i) << "  mappedValue=> "<< mappedValue << std::endl;
        if (input(i, dimension) == mappedValue)
        {
          Log::Info << "<<IMPUTER NON TRANSPOSE>>" << std::endl;
          output(i, dimension) = customValue;
        }
        else {
          Log::Info << "<not equal>" << std::endl;
        }
      }
    }
    Log::Info << "<CUSTOM imputer end>" << std::endl;
  }

}; // class Imputer

} // namespace data
} // namespace mlpack

#endif
