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
 *  - isSerializable: contains whether a parameter is a serializable model type
 *  - isHyperparam: contains whether a parameter is a hyperparameter (e.g. is a
 *      non-required scalar parameter to the _train binding)
 *  - isBool: contains whether a parameter is a boolean
 */
inline void ExtractGroupData(
    const std::string& groupName,
    const std::vector<std::string>& methods,
    std::map<std::string, mlpack::util::Params>& methodParams,
    std::string& trainBindingName,
    mlpack::util::ParamData*& modelType,
    std::vector<mlpack::util::ParamData*>& hyperparams);
inline void PopulateMethodMaps(
    const std::vector<std::string>& methods,
    std::map<std::string, mlpack::util::Params>& methodParams,
    std::map<std::string, bool>& isSerializable,
    std::map<std::string, bool>& isHyperparam,
    std::map<std::string, bool>& isBool);

} // namespace util
} // namespace bindings
} // namespace mlpack

#include "wrapper_utilities_impl.hpp"

#endif
