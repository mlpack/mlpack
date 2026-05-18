/**
 * @file bindings/util/wrapper_utilities.hpp
 * @author Nippun Sharma
 *
 * Utilities for generating wrapper classes in different binding languages.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_UTIL_WRAPPER_UTILITIES_HPP
#define MLPACK_BINDINGS_UTIL_WRAPPER_UTILITIES_HPP

#include <mlpack/base.hpp>

namespace mlpack {
namespace bindings {
namespace util {

// Given a space-separated list of methods, split them into a vector.
inline std::vector<std::string> GetMethods(const std::string& validMethods);

/**
 * Given a list of methods whose _main.cpp files have been included, extract
 * their parameters individually to populate each of these maps:
 *
 *  - methodParams: maps a method to its Params object
 *  - trainBindingName: the name of the binding that represents the train()
 *      function
 *  - modelType: the ParamData that represents the serializable model type
 *      (there should be only one)
 *  - hyperparams: a list of ParamData objects representing hyperparameters to
 *      the training binding
 */
inline void ExtractGroupData(
    const std::string& groupName,
    const std::vector<std::string>& methods,
    std::map<std::string, mlpack::util::Params>& methodParams,
    std::string& trainBindingName,
    mlpack::util::ParamData*& modelType,
    std::vector<mlpack::util::ParamData*>& hyperparams);

} // namespace util
} // namespace bindings
} // namespace mlpack

#include "wrapper_utilities_impl.hpp"

#endif
