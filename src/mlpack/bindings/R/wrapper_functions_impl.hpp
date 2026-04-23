/**
 * @file bindings/R/wrapper_functions_impl.hpp
 * @author Dirk Eddelbuettel
 *
 * Contains some utility functions for wrapper generation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_WRAPPER_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_R_WRAPPER_FUNCTIONS_IMPL_HPP

namespace mlpack {
namespace bindings {
namespace r {

inline std::string GetMappedName(const std::string& methodName)
{
  std::map<std::string, std::string> nameMap;
  nameMap["train"] = "train";
  nameMap["classify"] = "predict";
  nameMap["predict"] = "predict";
  nameMap["probabilities"] = "predict_proba";
  return nameMap[methodName];
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
