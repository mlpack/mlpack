/**
 * @file bindings/util/wrapper_utilities_impl.hpp
 * @author Nippun Sharma
 *
 * Implementation of language-agnostic utilities for generating binding wrapper
 * classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_UTIL_WRAPPER_UTILITIES_IMPL_HPP
#define MLPACK_BINDINGS_UTIL_WRAPPER_UTILITIES_IMPL_HPP

#include "wrapper_utilities.hpp"

namespace mlpack {
namespace bindings {
namespace util {

inline std::vector<std::string> GetMethods(const std::string& validMethods)
{
  std::vector<std::string> methods;
  std::stringstream methodStream(validMethods);
  std::string temp;

  while (std::getline(methodStream, temp, ' '))
  {
    methods.push_back(temp);
  }

  return methods;
}

inline void ExtractGroupData(
    const std::string& groupName,
    const std::vector<std::string>& methods,
    std::map<std::string, mlpack::util::Params>& methodParams,
    std::string& trainBindingName,
    mlpack::util::ParamData*& modelType,
    std::vector<mlpack::util::ParamData*>& hyperparams)
{
  for (const std::string& m : methods)
  {
    methodParams[m] = IO::Parameters(groupName + "_" + m);

    if (m == "train")
    {
      trainBindingName = m;
      // Now iterate over the training parameters and extract the
      // hyperparameters and serializable model type.  The training method must
      // emit a model.
      for (auto& it : methodParams[m].Parameters())
      {
        mlpack::util::ParamData& d = it.second;

        // Extract whether the parameter is a serializable model type.
        bool s;
        methodParams[m].functionMap[d.tname]["IsSerializable"](d, NULL,
            (void*) &s);
        if (s && modelType != nullptr)
        {
          throw std::runtime_error("ExtractGroupData(): binding '" + groupName +
              "_" + m + "' has multiple serializable model types!  Only one is "
              "allowed. Check INPUT_PARAM_*() and OUTPUT_PARAM_*() "
              "definitions.");
        }
        else if (s)
        {
          modelType = &d;
        }
        else if (d.input)
        {
          // It is not a serializable model, but it is an input parameter.  If
          // it's not an Armadillo object, then it must be a hyperparameter.
          if (d.cppType.find("arma") == std::string::npos)
            hyperparams.push_back(&d);
        }
      }

      if (modelType == nullptr)
      {
        throw std::runtime_error("ExtractGroupData(): binding '" + groupName +
            "_" + m + "' has no serializable model type!  Check "
            "INPUT_PARAM_*() and OUTPUT_PARAM_*() definitions.");
      }
    }
  }
}

inline void PopulateMethodMaps(
    const std::string& groupName,
    const std::vector<std::string>& methods,
    std::map<std::string, mlpack::util::Params>& methodParams,
    std::map<std::pair<std::string, std::string>, bool>& isSerializable,
    std::map<std::pair<std::string, std::string>, bool>& isHyperparam,
    std::map<std::pair<std::string, std::string>, bool>& isBool)
{
  for (const std::string& m : methods)
  {
    methodParams[m] = IO::Parameters(groupName + "_" + m);

    for (const auto& it : methodParams[m].Parameters())
    {
      const mlpack::util::ParamData& d = it.second;
      const std::pair<std::string, std::string> key = std::make_pair(m, d.name);

      // Extract whether the parameter is a serializable model type.
      bool s;
      methodParams[m].functionMap[d.tname]["IsSerializable"](
          (mlpack::util::ParamData&) d, NULL, (void*) &s);
      isSerializable[key] = s;

      // Extract whether the parameter is a boolean parameter.
      isBool[key] = (d.cppType == "bool");

      // If this is the training binding, then if the parameter is non-Armadillo
      // and non-serializable, then it is a hyperparameter.
      if (m.substr(m.size() - 6) == "_train")
      {
        const size_t foundArma = d.cppType.find("arma");
        if (foundArma == std::string::npos && !isSerializable[key])
          isHyperparam[key] = true;
        else
          isHyperparam[key] = false;
      }
      else
      {
        isHyperparam[std::make_pair(m, d.name)] = false;
      }
    }
  }
}

} // namespace util
} // namespace bindings
} // namespace mlpack

#endif
