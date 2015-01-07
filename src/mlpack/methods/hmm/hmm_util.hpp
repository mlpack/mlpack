/**
 * @file hmm_util.hpp
 * @author Ryan Curtin
 *
 * Save/load utilities for HMMs.  This should be eventually merged into the HMM
 * class itself.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_HMM_HMM_UTIL_HPP
#define __MLPACK_METHODS_HMM_HMM_UTIL_HPP

#include "hmm.hpp"

namespace mlpack {
namespace hmm {

/**
 * Save an HMM to file.  This only works for GMMs, DiscreteDistributions, and
 * GaussianDistributions.
 *
 * @tparam Distribution Distribution type of HMM.
 * @param sr SaveRestoreUtility to use.
 */
template<typename Distribution>
void SaveHMM(const HMM<Distribution>& hmm, util::SaveRestoreUtility& sr);

/**
 * Load an HMM from file.  This only works for GMMs, DiscreteDistributions, and
 * GaussianDistributions.
 *
 * @tparam Distribution Distribution type of HMM.
 * @param sr SaveRestoreUtility to use.
 */
template<typename Distribution>
void LoadHMM(HMM<Distribution>& hmm, util::SaveRestoreUtility& sr);

}; // namespace hmm
}; // namespace mlpack

// Include implementation.
#include "hmm_util_impl.hpp"

#endif
