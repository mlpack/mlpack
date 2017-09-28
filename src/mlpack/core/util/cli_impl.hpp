/**
 * @file cli_impl.hpp
 * @author Matthew Amidon
 *
 * Implementation of templated functions of the CLI class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_CLI_IMPL_HPP
#define MLPACK_CORE_UTIL_CLI_IMPL_HPP

// In case it has not already been included.
#include "cli.hpp"
#include "prefixedoutstream.hpp"

#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>

namespace mlpack {

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
  // Only use the alias if the parameter does not exist as given.
  std::string key =
      (GetSingleton().parameters.count(identifier) == 0 &&
       identifier.length() == 1 && GetSingleton().aliases.count(identifier[0]))
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

  // Do we have a special mapped function?
  if (CLI::GetSingleton().functionMap[d.tname].count("GetParam") != 0)
  {
    T* output = NULL;
    CLI::GetSingleton().functionMap[d.tname]["GetParam"](d, NULL,
        (void*) &output);
    return *output;
  }
  else
  {
    return *boost::any_cast<T>(&d.value);
  }
}

/**
 * Cast the given parameter of the given type to a short, printable std::string,
 * for use in status messages.  Ideally the message returned here should be only
 * a handful of characters, and certainly no longer than one line.
 *
 * @param identifier The name of the parameter in question.
 */
template<typename T>
std::string CLI::GetPrintableParam(const std::string& identifier)
{
  // Only use the alias if the parameter does not exist as given.
  std::string key = ((GetSingleton().parameters.count(identifier) == 0) &&
      (identifier.length() == 1) &&
      (GetSingleton().aliases.count(identifier[0]) > 0)) ?
      GetSingleton().aliases[identifier[0]] : identifier;

  if (GetSingleton().parameters.count(key) == 0)
    Log::Fatal << "Parameter --" << key << " does not exist in this program!"
        << std::endl;

  util::ParamData& d = GetSingleton().parameters[key];

  // Make sure the types are correct.
  if (TYPENAME(T) != d.tname)
    Log::Fatal << "Attempted to access parameter --" << key << " as type "
        << TYPENAME(T) << ", but its true type is " << d.tname << "!"
        << std::endl;

  // Do we have a special mapped function?
  if (CLI::GetSingleton().functionMap[d.tname].count("GetPrintableParam") != 0)
  {
    std::string output;
    CLI::GetSingleton().functionMap[d.tname]["GetPrintableParam"](d, NULL,
        (void*) &output);
    return output;
  }
  else
  {
    std::ostringstream oss;
    oss << "no GetPrintableParam function handler registered for type "
        << d.cppType;
    throw std::runtime_error(oss.str());
  }
}

template<typename T>
T& CLI::GetRawParam(const std::string& identifier)
{
  // Only use the alias if the parameter does not exist as given.
  std::string key =
      (GetSingleton().parameters.count(identifier) == 0 &&
       identifier.length() == 1 && GetSingleton().aliases.count(identifier[0]))
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

  // Do we have a special mapped function?
  if (CLI::GetSingleton().functionMap[d.tname].count("GetRawParam") != 0)
  {
    T* output = NULL;
    CLI::GetSingleton().functionMap[d.tname]["GetRawParam"](d, NULL,
        (void*) &output);
    return *output;
  }
  else
  {
    // Use the regular GetParam().
    return GetParam<T>(identifier);
  }
}

} // namespace mlpack

#endif
