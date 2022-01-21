/**
 * @file params_impl.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Implementation of functions in the Params class.
 */
#ifndef MLPACK_CORE_UTIL_PARAMS_IMPL_HPP
#define MLPACK_CORE_UTIL_PARAMS_IMPL_HPP

// Include definition, if needed.
#include "params.hpp"

namespace mlpack {
namespace util {

/**
 * Get the value of type T found for the parameter specified by `identifier`.
 * You can set the value using this reference safely.
 *
 * @param identifier The name of the parameter in question.
 */
template<typename T>
T& Params::Get(const std::string& identifier)
{
  // TODO: can we remove the alias support here?
  // Only use the alias if the parameter does not exist as given.
  std::string key = (parameters.count(identifier) == 0 &&
      identifier.length() == 1 && aliases.count(identifier[0])) ?
      aliases[identifier[0]] : identifier;

  if (parameters.count(key) == 0)
    Log::Fatal << "Parameter '" << key << "' does not exist in this program!"
        << std::endl;

  ParamData& d = parameters[key];

  // Make sure the types are correct.
  if (TYPENAME(T) != d.tname)
    Log::Fatal << "Attempted to access parameter '" << key << "' as type "
        << TYPENAME(T) << ", but its true type is " << d.tname << "!"
        << std::endl;

  // Do we have a special mapped function?
  if (functionMap[d.tname].count("GetParam") != 0)
  {
    T* output = NULL;
    functionMap[d.tname]["GetParam"](d, NULL, (void*) &output);
    return *output;
  }
  else
  {
    return *ANY_CAST<T>(&d.value);
  }
}

/**
 * Cast the given parameter of the given type to a short, printable
 * `std::string`, for use in status messages.  The message returned here
 * should be only a handful of characters, and certainly no longer than one
 * line.
 *
 * @param identifier The name of the parameter in question.
 */
template<typename T>
std::string Params::GetPrintable(const std::string& identifier)
{
  // TODO: can we remove the alias support here?
  // Only use the alias if the parameter does not exist as given.
  std::string key = ((parameters.count(identifier) == 0) &&
      (identifier.length() == 1) && (aliases.count(identifier[0]) > 0)) ?
      aliases[identifier[0]] : identifier;

  if (parameters.count(key) == 0)
    Log::Fatal << "Parameter '" << key << "' does not exist in this program!"
        << std::endl;

  ParamData& d = parameters[key];

  // Make sure the types are correct.
  if (TYPENAME(T) != d.tname)
    Log::Fatal << "Attempted to access parameter '" << key << "' as type "
        << TYPENAME(T) << ", but its true type is " << d.tname << "!"
        << std::endl;

  // Do we have a special mapped function?
  if (functionMap[d.tname].count("GetPrintableParam") != 0)
  {
    std::string output;
    functionMap[d.tname]["GetPrintableParam"](d, NULL, (void*) &output);
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

/**
 * Get the raw value of the parameter before any processing that Get() might
 * normally do.  So, e.g., for command-line programs, this does not
 * perform any data loading or manipulation like Get() does.  So if you
 * want to access a matrix or model (or similar) parameter before it is
 * loaded, this is the method to use.
 *
 * @param identifier The name of the parameter in question.
 */
template<typename T>
T& Params::GetRaw(const std::string& identifier)
{
  // TODO: can we remove the alias support here?
  // Only use the alias if the parameter does not exist as given.
  std::string key = (parameters.count(identifier) == 0 &&
      identifier.length() == 1 && aliases.count(identifier[0])) ?
      aliases[identifier[0]] : identifier;

  if (parameters.count(key) == 0)
    Log::Fatal << "Parameter '" << key << "' does not exist in this program!"
        << std::endl;

  ParamData& d = parameters[key];

  // Make sure the types are correct.
  if (TYPENAME(T) != d.tname)
    Log::Fatal << "Attempted to access parameter '" << key << "' as type "
        << TYPENAME(T) << ", but its true type is " << d.tname << "!"
        << std::endl;

  // Do we have a special mapped function?
  if (functionMap[d.tname].count("GetRawParam") != 0)
  {
    T* output = NULL;
    functionMap[d.tname]["GetRawParam"](d, NULL, (void*) &output);
    return *output;
  }
  else
  {
    // Use the regular GetParam().
    return Get<T>(identifier);
  }
}

//! Utility function, used by CheckInputMatrices().
template<typename T>
void Params::CheckInputMatrix(const T& matrix, const std::string& identifier)
{
  const std::string errMsg1 = "The input '" + identifier + "' has NaN values.";
  const std::string errMsg2 = "The input '" + identifier + "' has inf values.";

  if (matrix.has_nan())
    Log::Fatal << errMsg1 << std::endl;
  if (matrix.has_inf())
    Log::Fatal << errMsg2 << std::endl;
}

} // namespace util
} // namespace mlpack

#endif
