/**
 * @file bindings/julia/wrapper_functions_impl.hpp
 * @author Ryan Curtin
 *
 * Contains some important utility functions for wrapper generation for Julia.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_WRAPPER_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_JULIA_WRAPPER_FUNCTIONS_IMPL_HPP

namespace mlpack {
namespace bindings {
namespace julia {

inline std::string GetClassName(const std::string& groupName)
{
  std::string className = "";
  std::stringstream groupNameStream(groupName);
  std::string temp;

  while (std::getline(groupNameStream, temp, '_'))
  {
    temp[0] = std::toupper(temp[0]);
    className += temp;
  }

  return className;
}

inline std::string GetValidName(const std::string& paramName)
{
  // There is no special handling of parameter names in Julia.
  return paramName;
}

inline std::string GetMappedName(const std::string& methodName)
{
  if (methodName == "train")
    return "fit!";
  else if (methodName == "classify")
    return "predict";
  else if (methodName == "probabilities")
    return "predict_proba";
  else
    return methodName;
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
