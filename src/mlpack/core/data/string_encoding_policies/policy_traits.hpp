/**
 * @file core/data/string_encoding_policies/policy_traits.hpp
 * @author Jeffin Sam
 * @author Mikhail Lozhnikov
 *
 * This provides the StringEncodingPolicyTraits struct, a template struct to
 * get information about various encoding policies.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_POLICY_TRAITS_HPP
#define MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_POLICY_TRAITS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

/**
 * This is a template struct that provides some information about various
 * encoding policies.
 */
template<class PolicyType>
struct StringEncodingPolicyTraits
{
  /**
   * Indicates if the policy is able to encode the token at once without
   * any information about other tokens as well as the total tokens count.
   */
  static const bool onePassEncoding = false;
};

} // namespace data
} // namespace mlpack

#endif
