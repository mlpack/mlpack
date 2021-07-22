/**
 * @file bindings/cli/parameter_type.hpp
 * @author Ryan Curtin
 *
 * Template metaprogramming structures to find the type (as seen by
 * CLI11) of a particular option type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_PARAMETER_TYPE_HPP
#define MLPACK_BINDINGS_CLI_PARAMETER_TYPE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

// Default: HasSerialize = false.
template<bool HasSerialize, typename T>
struct ParameterTypeDeducer
{
  typedef T type;
};

// If we have a serialize() function, then the type is a string.
template<typename T>
struct ParameterTypeDeducer<true, T>
{
  typedef std::string type;
};

/**
 * Utility struct to return the type that CLI11 should accept
 * for a given input type.  In general, there is no change from the input type,
 * but in some cases this may be another type.
 */
template<typename T>
struct ParameterType
{
  typedef typename ParameterTypeDeducer<data::HasSerialize<T>::value, T>::type
      type;
};

/**
 * For vector types, CLI11 will accept a std::string, not an
 * arma::Col<eT> (since it is not clear how to specify a vector on the
 * command-line).
 */
template<typename eT>
struct ParameterType<arma::Col<eT>>
{
  typedef std::tuple<std::string, size_t, size_t> type;
};

/**
 * For row vector types, CLI11 will accept a std::string, not
 * an
 * arma::Row<eT> (since it is not clear how to specify a vector on the
 * command-line).
 */
template<typename eT>
struct ParameterType<arma::Row<eT>>
{
  typedef std::tuple<std::string, size_t, size_t> type;
};

/**
 * For matrix types, CLI11 will accept a std::string, not an
 * arma::mat (since it is not clear how to specify a matrix on the
 * command-line).
 */
template<typename eT>
struct ParameterType<arma::Mat<eT>>
{
  typedef std::tuple<std::string, size_t, size_t> type;
};

/**
 * For matrix+dataset info types, we should accept a std::string.
 */
template<typename eT, typename PolicyType>
struct ParameterType<std::tuple<mlpack::data::DatasetMapper<PolicyType,
                         std::string>, arma::Mat<eT>>>
{
  typedef std::tuple<std::string, size_t, size_t> type;
};

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
