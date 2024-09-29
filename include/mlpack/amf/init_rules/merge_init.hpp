/**
 * @file methods/amf/init_rules/merge_init.hpp
 * @author Ziyang Jiang
 *
 * Initialization rule for alternating matrix factorization (AMF). This simple
 * initialization is performed by assigning a given matrix to W or H and a 
 * random matrix to another one.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AMF_MERGE_INIT_HPP
#define MLPACK_METHODS_AMF_MERGE_INIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This initialization rule for AMF simply takes in two initialization rules,
 * and initialize W with the first rule and H with the second rule.
 */
template<typename WInitializationRuleType, typename HInitializationRuleType>
class MergeInitialization
{
 public:
  // Empty constructor required for the InitializeRule template
  MergeInitialization() { }

  // Initialize the MergeInitialization object with existing initialization
  // rules.
  MergeInitialization(const WInitializationRuleType& wInitRule,
                      const HInitializationRuleType& hInitRule) :
                      wInitializationRule(wInitRule),
                      hInitializationRule(hInitRule)
  { }

  /**
   * Initialize W and H with the corresponding initialization rules.
   *
   * @param V Input matrix.
   * @param r Rank of decomposition.
   * @param W W matrix, to be initialized to given matrix.
   * @param H H matrix, to be initialized to given matrix.
   */
  template<typename MatType, typename WHMatType>
  inline void Initialize(const MatType& V,
                         const size_t r,
                         WHMatType& W,
                         WHMatType& H)
  {
    wInitializationRule.InitializeOne(V, r, W);
    hInitializationRule.InitializeOne(V, r, H, false);
  }

 private:
  // Initialization rule for W matrix
  WInitializationRuleType wInitializationRule;
  // Initialization rule for H matrix
  HInitializationRuleType hInitializationRule;
};

} // namespace mlpack

#endif
