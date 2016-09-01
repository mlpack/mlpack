/**
 * @file cli_impl.hpp
 * @author Matthew Amidon
 *
 * Implementation of templated functions of the CLI class.
 */
#ifndef MLPACK_CORE_UTIL_CLI_IMPL_HPP
#define MLPACK_CORE_UTIL_CLI_IMPL_HPP

// In case it has not already been included.
#include "cli.hpp"
#include "prefixedoutstream.hpp"

// Include option.hpp here because it requires CLI but is also templated.
#include "option.hpp"

#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>

namespace mlpack {

/**
 * @brief Adds a parameter to CLI, making it accessibile via GetParam &
 *     CheckValue.
 *
 * @tparam T The type of the parameter.
 * @param identifier The name of the parameter, eg foo.
 * @param description A string description of the parameter.
 * @param alias Short name of the parameter.
 * @param required If required, the program will refuse to run unless the
 *     parameter is specified.
 * @param input If true, the parameter is an input parameter (not an output
 *     parameter).
 * @param noTranspose If true and the parameter is a matrix type, then the
 *     matrix will not be transposed when it is loaded.
 */
template<typename T>
void CLI::Add(const std::string& identifier,
              const std::string& description,
              const char alias,
              const bool required,
              const bool input,
              const bool noTranspose)
{
  // Temporarily define color code escape sequences.
  #ifndef _WIN32
    #define BASH_RED "\033[0;31m"
    #define BASH_CLEAR "\033[0m"
  #else
    #define BASH_RED ""
    #define BASH_CLEAR ""
  #endif

  // Temporary outstream object for detecting duplicate identifiers.
  util::PrefixedOutStream outstr(std::cerr,
        BASH_RED "[FATAL] " BASH_CLEAR, false, true /* fatal */);

  #undef BASH_RED
  #undef BASH_CLEAR

  // Define identifier and alias maps.
  std::map<std::string, util::ParamData>& parameters =
      GetSingleton().parameters;
  std::map<char, std::string>& aliases = GetSingleton().aliases;

  // If found in current map, print fatal error and terminate the program.
  if (parameters.count(identifier))
    outstr << "Parameter --" << identifier << " (-" << alias << ") "
           << "is defined multiple times with the same identifiers."
           << std::endl;
  if (alias != '\0' && aliases.count(alias))
    outstr << "Parameter --" << identifier << " (-" << alias << ") "
           << "is defined multiple times with the same alias." << std::endl;

  // Add the parameter to the map of parameters.  We must hold two boost::any
  // types, in case the type that boost::po receives will be different than the
  // type we want (like for matrices).
  util::ParamData data;
  typename util::ParameterType<T>::type tmp;
  T mappedTmp;

  data.desc = description;
  data.name = identifier;
  data.tname = TYPENAME(T);
  data.alias = alias;
  data.isFlag = (TYPENAME(T) == TYPENAME(bool));
  data.noTranspose = noTranspose;
  data.required = required;
  data.input = input;
  data.loaded = false;
  data.value = boost::any(tmp);
  data.boostName = util::MapParameterName<T>(identifier);
  data.mappedValue = boost::any(mappedTmp);

  // Sanity check: ensure that the boost name is not already in use.
  std::map<std::string, util::ParamData>::const_iterator it =
      parameters.begin();
  while (it != parameters.end())
  {
    if ((*it).second.boostName == data.boostName ||
        (*it).second.name == data.boostName)
      outstr << "Parameter --" << data.boostName << " (" << alias << ") "
             << "is defined multiple times with the same identifiers."
             << std::endl;
    ++it;
  }

  GetSingleton().parameters[identifier] = data;

  // Now add the parameter name to boost::program_options.
  po::options_description& desc = CLI::GetSingleton().desc;
  // Must make use of boost syntax here.
  std::string progOptId = (alias != '\0') ? data.boostName + ","
      + std::string(1, alias) : data.boostName;

  // Add the alias, if necessary.
  if (alias != '\0')
    GetSingleton().aliases[alias] = identifier;

  // Add the option to boost program_options.
  if (data.isFlag)
    desc.add_options()(progOptId.c_str(), description.c_str());
  else
    desc.add_options()(progOptId.c_str(),
        po::value<typename util::ParameterType<T>::type>(),
        description.c_str());

  // If the option is required, add it to the required options list.
  if (required)
    GetSingleton().requiredOptions.push_front(identifier);

  // Depending on whether or not the option is input or output, add it to the
  // appropriate list.
  if (!input)
    GetSingleton().outputOptions.push_front(identifier);
}

/**
 * @brief Returns the value of the specified parameter.
 *   If the parameter is unspecified, an undefined but
 *   more or less valid value is returned.
 *
 * @tparam T The type of the parameter.
 * @param identifier The full name of the parameter.
 *
 * @return The value of the parameter.  Use CLI::CheckValue to determine if it's
 *     valid.
 */
template<typename T>
T& CLI::GetParam(const std::string& identifier)
{
  std::string key =
      (identifier.length() == 1 && GetSingleton().aliases.count(identifier[0]))
      ? GetSingleton().aliases[identifier[0]] : identifier;

  if (GetSingleton().parameters.count(key) == 0)
    Log::Fatal << "Parameter --" << key << " does not exist in this program!"
        << std::endl;

  util::ParamData& d = GetSingleton().parameters[key];

  // Make sure the types are correct.
  if (TYPENAME(T) != d.tname)
    Log::Fatal << "Attempted to access parameter --" << key << " as type "
        << TYPENAME(T) << ", but its true type is " << d.tname << "!"
        << std::endl;

  // We already know that required options have been passed, so we have a valid
  // value to return.  Because the parameters held are sometimes different types
  // than what the user wants, we must pass through a utility function.
  typename util::ParameterType<T>::type& v =
      *boost::any_cast<typename util::ParameterType<T>::type>(&d.value);
  return util::HandleParameter<T>(v, d);
}

//! This overload is called when nothing special needs to happen to the name of
//! the parameter.
template<typename T>
std::string util::MapParameterName(
    const std::string& identifier,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */)
{
  return identifier;
}

//! This overload is called when T == ParameterType<T>::value.  An overload for
//! matrices is in cli.cpp.
template<typename T>
T& util::HandleParameter(
    typename util::ParameterType<T>::type& value,
    util::ParamData& /* d */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */)
{
  return value;
}

//! This is called for matrices, which have a different boost name.
template<typename T>
std::string util::MapParameterName(
    const std::string& identifier,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* /* junk */)
{
  return identifier + "_file";
}

//! This overload is called for matrices, which return a different type.
template<typename T>
T& util::HandleParameter(
    typename util::ParameterType<T>::type& value,
    util::ParamData& d,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* /* junk */)
{
  // If the matrix is an input matrix, we have to load the matrix.  'value'
  // contains the filename.  It's possible we could load empty matrices many
  // times, but I am not bothered by that---it shouldn't be something that
  // happens.
  T& matrix = *boost::any_cast<T>(&d.mappedValue);
  if (d.input && !d.loaded)
  {
    data::Load(value, matrix, true, !d.noTranspose);
    d.loaded = true;
  }

  return matrix;
}


} // namespace mlpack

#endif
