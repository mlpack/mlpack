/**
 * @file policy_traits.hpp
 * @author Jeffin Sam
 *
 * This provides the PolicyTraits strucutre, a template class to get information
 * about various Encoding Policies.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_POLICY_TRAITS_HPP
#define MLPACK_CORE_DATA_POLICY_TRAITS_HPP

#include <mlpack/prereqs.hpp>
#include <utility>

namespace mlpack {
namespace data {

template<class Policy>
struct PolicyTraits
{
  /**
  * This is used to indicate whether a policy has a function which can
  * have output without any padding of zroes.
  */
  static const bool outputWithNoPadding = false;
};

} // namespace data
} // namespace mlpack

#endif
