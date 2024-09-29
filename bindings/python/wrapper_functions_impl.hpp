/**
 * @file bindings/python/wrapper_functions_impl.hpp
 * @author Nippun Sharma
 *
 * Contains some important utility functions for wrapper generation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_WRAPPER_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_PYTHON_WRAPPER_FUNCTIONS_IMPL_HPP

namespace mlpack {
namespace bindings {
namespace python {

inline std::string GetClassName(const std::string& groupName)
{
  std::string className = "";
  std::stringstream groupNameStream(groupName);
  std::string temp;

  while(std::getline(groupNameStream, temp, '_'))
  {
    temp[0] = std::toupper(temp[0]);
    className += temp;
  }

  return className;
}

inline std::string GetValidName(const std::string& paramName)
{
  std::string correctParamName;

  if (paramName == "lambda")
    correctParamName = "lambda_";
  else if (paramName == "input")
    correctParamName = "input_";
  else
    correctParamName = paramName;

  return correctParamName;
}

inline std::vector<std::string> GetMethods(const std::string& validMethods)
{
  std::vector<std::string> methods;
  std::stringstream methodStream(validMethods);
  std::string temp;

  while(std::getline(methodStream, temp, ' '))
  {
    methods.push_back(temp);
  }

  return methods;
}

inline std::string GetMappedName(const std::string& methodName)
{
  std::map<std::string, std::string> nameMap;
  nameMap["train"] = "fit";
  nameMap["classify"] = "predict";
  nameMap["predict"] = "predict";
  nameMap["probabilities"] = "predict_proba";
  return nameMap[methodName];
}

} // python.
} // bindings.
} // mlpack.

#endif
