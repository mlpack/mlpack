/**
 * @file param_data.hpp
 * @author Ryan Curtin
 *
 * This defines the structure that holds information for each command-line
 * parameter, as well as utility functions it is used with.
 */
#ifndef MLPACK_CORE_UTIL_PARAM_DATA_HPP
#define MLPACK_CORE_UTIL_PARAM_DATA_HPP

#include <mlpack/prereqs.hpp>

#include <boost/any.hpp>

/**
 * The TYPENAME macro is used internally to convert a type into a string.
 */
#define TYPENAME(x) (std::string(typeid(x).name()))

namespace mlpack {
namespace util {

//! Metaprogramming structure for vector detection.
template<typename T>
struct IsStdVector { const static bool value = false; };

//! Metaprogramming structure for vector detection.
template<typename T, typename A>
struct IsStdVector<std::vector<T, A>> { const static bool value = true; };

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

/**
 * This structure holds all of the information about a single parameter,
 * including its value (which is set when ParseCommandLine() is called).  It
 * does not hold any information about whether or not it was passed---that is
 * handled elsewhere.  A ParamData struct is only useful in order to get
 * "static" information about a parameter.  Note that some parameter types have
 * internal types but also different types that are used by
 * boost::program_options (specifically, matrix and model types map to strings).
 *
 * This structure is somewhat unwieldy and is likely to be refactored at some
 * point in the future, but for now it does the job fine.
 */
struct ParamData
{
  //! Name of this parameter.  This is the name used for HasParam() and
  //! GetParam().
  std::string name;
  //! Description of this parameter, if any.
  std::string desc;
  //! Type information of this parameter.  Note that this is TYPENAME() of the
  //! user-visible parameter type, not whatever is given by ParameterType<>.
  std::string tname;
  //! Alias for this parameter.
  char alias;
  //! True if the wasPassed value should not be ignored.
  bool isFlag;
  //! True if this is a matrix that should not be transposed.  Ignored if the
  //! parameter is not a matrix.
  bool noTranspose;
  //! True if this option is required.
  bool required;
  //! True if this option is an input option (otherwise, it is output).
  bool input;
  //! If this is an input parameter that needs extra loading, this indicates
  //! whether or not it has been loaded.
  bool loaded;
  //! If this is a matrix or model parameter, then boost::program_options will
  //! actually represent this as a string.
  bool isMappedString;
  //! The actual value that is held, as passed from the user (so the type could
  //! be different than the type of the parameter).
  boost::any value;
  //! The value that the user interacts with, if the type is different than the
  //! type of the parameter.  This is used to store matrices, for instance,
  //! because 'value' must hold the string name that the user passed.
  boost::any mappedValue;
  //! The name of the parameter, as seen by boost::program_options.
  std::string boostName;
  //! When the CLI object is destructed, output options must be output.  If
  //! 'input' is false, then this function pointer should point to a function
  //! that outputs the parameter.
  void (*outputFunction)(const ParamData&);
  //! When the CLI object is destructed, output a string representation of the
  //! parameter.
  void (*printFunction)(const ParamData&);
  //! When the CLI object is asked to print help, output the default value.
  std::string (*defaultFunction)(const ParamData&);
  //! When the CLI object is asked to print parameter types, output a string
  //! version of this parameter's type.
  std::string (*stringTypeFunction)();
};

/**
 * If needed, map the parameter name to the name that is used by boost.  This
 * is generally the same as the name, but for matrices it may be different.
 */
template<typename T>
std::string MapParameterName(
    const std::string& identifier,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0);

//! This must be overloaded for matrices.
template<typename T>
std::string MapParameterName(
    const std::string& identifier,
    const typename boost::enable_if_c<
        arma::is_arma_type<T>::value ||
        std::is_same<T, std::tuple<mlpack::data::DatasetInfo,
                                   arma::mat>>::value ||
        data::HasSerialize<T>::value>::type* /* junk */ = 0);

/**
 * If needed, map 'trueValue' to the right type and return it.  This is called
 * from GetParam().
 */
template<typename T>
T& HandleParameter(
    typename util::ParameterType<T>::type& value,
    util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0);

//! This must be overloaded for matrices.
template<typename T>
T& HandleParameter(
    typename util::ParameterType<T>::type& value,
    util::ParamData& d,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0);

//! This must be overloaded for matrices and dataset info objects.
template<typename T>
T& HandleParameter(
    typename util::ParameterType<T>::type& value,
    util::ParamData& d,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0);

//! This must be overloaded for serializable objects.
template<typename T>
T& HandleParameter(
    typename util::ParameterType<T>::type& value,
    util::ParamData& d,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);

//! This will just return the value.
template<typename T>
T& HandleRawParameter(
    typename util::ParameterType<T>::type& value,
    util::ParamData& /* d */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  return value;
}

//! This will return the mapped value.
template<typename T>
T& HandleRawParameter(
    typename util::ParameterType<T>::type& /* value */,
    util::ParamData& d,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  return *boost::any_cast<T>(&d.mappedValue);
}

//! This will return the mapped value.
template<typename T>
T& HandleRawParameter(
    typename util::ParameterType<T>::type& /* value */,
    util::ParamData& d,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  return *boost::any_cast<T>(&d.mappedValue);
}

//! This will return the mapped value.
template<typename T>
T& HandleRawParameter(
    typename util::ParameterType<T>::type& /* value */,
    util::ParamData& d,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  return *boost::any_cast<T>(&d.mappedValue);
}

} // namespace util
} // namespace mlpack

//! Include implementation of functions.
#include "param_data_impl.hpp"

#endif
