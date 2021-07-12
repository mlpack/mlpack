/**
 * @file params.cpp
 * @author Ryan Curtin
 *
 * Implementation of functions in the Param class.
 */
#include "params.hpp"
#include <mlpack/core/data/dataset_mapper.hpp>

namespace mlpack {
namespace util {

Params::Params(const std::map<char, std::string>& aliases,
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

Params::Params()
{
  // Nothing to do.
}

/**
 * Return `true` if the specified parameter was given.
 *
 * @param identifier The name of the parameter in question.
 */
bool Params::Has(const std::string& key) const
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
 * Given two (matrix) parameters, ensure that the first is an in-place copy of
 * the second.  This will generally do nothing (as the bindings already do
 * this automatically), except for command-line bindings, where we need to
 * ensure that the output filename is the same as the input filename.
 *
 * @param outputParamName Name of output (matrix) parameter.
 * @param inputParamName Name of input (matrix) parameter.
 */
void Params::MakeInPlaceCopy(const std::string& outputParamName,
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
void Params::SetPassed(const std::string& name)
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
void Params::CheckInputMatrices()
{
  typedef typename std::tuple<data::DatasetInfo, arma::mat> TupleType;
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
      CheckInputMatrix(std::get<1>(Get<TupleType>(paramName)), paramName);
    }
  }
}

} // namespace util
} // namespace mlpack
