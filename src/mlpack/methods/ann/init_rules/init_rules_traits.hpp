/**
 * @file methods/ann/init_rules/init_rules_traits.hpp
 * @author Marcus Edel
 *
 * This provides the InitTraits class, a template class to get information
 * about various initialization methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_INIT_RULES_TRAITS_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_INIT_RULES_TRAITS_HPP

namespace mlpack {

/**
 * This is a template class that can provide information about various
 * initialization methods. By default, this class will provide the weakest
 * possible assumptions on the initialization method, and each initialization
 * method should override values as necessary. If a initialization method
 * doesn't need to override a value, then there's no need to write a InitTraits
 * specialization for that class.
 */
template<typename InitRuleType>
class InitTraits
{
 public:
  /**
   * This is true if the initialization method is used for a single layer.
   */
  static const bool UseLayer = true;
};

} // namespace mlpack

#endif
