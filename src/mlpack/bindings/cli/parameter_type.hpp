/**
 * @file parameter_type.hpp
 * @author Ryan Curtin
 *
 * Template metaprogramming structures to find the type (as seen by
 * boost::program_options) of a particular option type.
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

// If we have a Serialize() function, then the type is a string.
template<typename T>
struct ParameterTypeDeducer<true, T>
{
  typedef std::string type;
};

/**
 * Utility struct to return the type that boost::program_options should accept
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
 * For vector types, boost::program_options will accept a std::string, not an
 * arma::Col<eT> (since it is not clear how to specify a vector on the
 * command-line).
 */
template<typename eT>
struct ParameterType<arma::Col<eT>>
{
  typedef std::string type;
};

/**
 * For row vector types, boost::program_options will accept a std::string, not
 * an
 * arma::Row<eT> (since it is not clear how to specify a vector on the
 * command-line).
 */
template<typename eT>
struct ParameterType<arma::Row<eT>>
{
  typedef std::string type;
};

/**
 * For matrix types, boost::program_options will accept a std::string, not an
 * arma::mat (since it is not clear how to specify a matrix on the
 * command-line).
 */
template<typename eT>
struct ParameterType<arma::Mat<eT>>
{
  typedef std::string type;
};

/**
 * For matrix+dataset info types, we should accept a std::string.
 */
template<typename eT, typename PolicyType>
struct ParameterType<std::tuple<mlpack::data::DatasetMapper<PolicyType>,
                     arma::Mat<eT>>>
{
  typedef std::string type;
};

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
