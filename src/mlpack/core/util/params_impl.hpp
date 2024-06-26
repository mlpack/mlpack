/**
 * @file params_impl.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Implementation of functions in the Params class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_PARAMS_IMPL_HPP
#define MLPACK_CORE_UTIL_PARAMS_IMPL_HPP

#include "log.hpp"

// Include definition, if needed.
#include "forward.hpp"
#include "params.hpp"

namespace mlpack {
namespace util {

inline Params::Params(const std::map<char, std::string>& aliases,
                      const std::map<std::string, ParamData>& parameters,
                      Params::FunctionMapType& functionMap,
                      const std::string& bindingName,
                      const BindingDetails& doc) :
    // Copy all the given inputs.
    aliases(aliases),
    parameters(parameters),
    functionMap(functionMap),
    bindingName(bindingName),
    doc(doc)
{
  // Nothing to do.
}

inline Params::Params()
{
  // Nothing to do.
}

/**
 * Return `true` if the specified parameter was given.
 *
 * @param key The name of the parameter in question.
 */
inline bool Params::Has(const std::string& key) const
{
  std::string usedKey = key;

  if (!parameters.count(key))
  {
    // Check any aliases, but only after we are sure the actual option as given
    // does not exist.
    // TODO: can we isolate alias support inside of the CLI binding code?
    if (key.length() == 1 && aliases.count(key[0]))
      usedKey = aliases.at(key[0]);

    if (!parameters.count(usedKey))
    {
      Log::Fatal << "Parameter '" << key << "' does not exist in this "
          << "program." << std::endl;
    }
  }
  const std::string& checkKey = usedKey;

  return (parameters.at(checkKey).wasPassed > 0);
}

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
    return *std::any_cast<T>(&d.value);
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

/**
 * Given two (matrix) parameters, ensure that the first is an in-place copy of
 * the second.  This will generally do nothing (as the bindings already do
 * this automatically), except for command-line bindings, where we need to
 * ensure that the output filename is the same as the input filename.
 *
 * @param outputParamName Name of output (matrix) parameter.
 * @param inputParamName Name of input (matrix) parameter.
 */
inline void Params::MakeInPlaceCopy(const std::string& outputParamName,
                                    const std::string& inputParamName)
{
  if (!parameters.count(outputParamName))
    Log::Fatal << "Unknown parameter '" << outputParamName << "'!" << std::endl;
  if (!parameters.count(inputParamName))
    Log::Fatal << "Unknown parameter '" << inputParamName << "'!" << std::endl;

  ParamData& output = parameters[outputParamName];
  ParamData& input = parameters[inputParamName];

  if (output.cppType != input.cppType)
  {
    Log::Fatal << "Cannot call MakeInPlaceCopy() with different types ("
        << output.cppType << " and " << input.cppType << ")!" << std::endl;
  }

  // Is there a function to do this?
  if (functionMap[output.tname].count("InPlaceCopy") != 0)
  {
    functionMap[output.tname]["InPlaceCopy"](output, (void*) &input, NULL);
  }
}

/**
 * Set the particular parameter as passed.
 *
 * @param identifier The name of the parameter to set as passed.
 */
inline void Params::SetPassed(const std::string& name)
{
  if (parameters.count(name) == 0)
  {
    throw std::invalid_argument("Params::SetPassed(): parameter " + name +
        " not known for binding " + bindingName + "!");
  }

  // Set passed to true.
  parameters[name].wasPassed = true;
}

/**
 * Check all input matrices for NaN and inf values, and throw an exception if
 * any are found.
 */
inline void Params::CheckInputMatrices()
{
  std::map<std::string, ParamData>::iterator itr;

  for (itr = parameters.begin(); itr != parameters.end(); ++itr)
  {
    std::string paramName = itr->first;
    std::string paramType = itr->second.cppType;
    if (paramType == "arma::mat")
    {
      CheckInputMatrix(Get<arma::mat>(paramName), paramName);
    }
    else if (paramType == "arma::vec")
    {
      CheckInputMatrix(Get<arma::vec>(paramName), paramName);
    }
    else if (paramType == "arma::rowvec")
    {
      CheckInputMatrix(Get<arma::rowvec>(paramName), paramName);
    }
    else if (paramType == "std::tuple<mlpack::data::DatasetInfo, arma::mat>")
    {
      // Note that CheckCategoricalParam() is a utility function that must be
      // defined after DatasetInfo is fully defined.
      data::CheckCategoricalParam(*this, paramName);
    }
  }
}

} // namespace util
} // namespace mlpack

#endif
