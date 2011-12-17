/**
 * @file hmm_util.hpp
 * @author Ryan Curtin
 *
 * Save/load utilities for HMMs.  This should be eventually merged into the HMM
 * class itself.
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
void SaveHMM(const HMM<Distribution>& hmm, utilities::SaveRestoreUtility& sr);

/**
 * Load an HMM from file.  This only works for GMMs, DiscreteDistributions, and
 * GaussianDistributions.
 *
 * @tparam Distribution Distribution type of HMM.
 * @param sr SaveRestoreUtility to use.
 */
template<typename Distribution>
void LoadHMM(HMM<Distribution>& hmm, utilities::SaveRestoreUtility& sr);

}; // namespace hmm
}; // namespace mlpack

// Include implementation.
#include "hmm_util_impl.hpp"

#endif
