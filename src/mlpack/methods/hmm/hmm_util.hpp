/**
 * @file hmm_util.hpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Deprecated Save/load utilities for HMMs. See HMM::Save, HMM::Load.
 */
#ifndef __MLPACK_METHODS_HMM_HMM_UTIL_HPP
#define __MLPACK_METHODS_HMM_HMM_UTIL_HPP

#include "hmm.hpp"

namespace mlpack {
namespace hmm {

/**
 * Save an HMM to file (deprecated).
 *
 * @tparam Distribution Distribution type of HMM.
 * @param sr SaveRestoreUtility to use.
 */
template<typename Distribution>
void SaveHMM(const HMM<Distribution>& hmm, util::SaveRestoreUtility& sr);

/**
 * Load an HMM from file (deprecated).
 *
 * @tparam Distribution Distribution type of HMM.
 * @param sr SaveRestoreUtility to use.
 */
template<typename Distribution>
void LoadHMM(HMM<Distribution>& hmm, util::SaveRestoreUtility& sr);

/**
 * Converter for HMMs saved using older MLPACK versions.
 *
 * @tparam Distribution Distribution type of HMM.
 * @param sr SaveRestoreUtility to use.
 */
template<typename Distribution>
void ConvertHMM(HMM<Distribution>& hmm, const util::SaveRestoreUtility& sr);

}; // namespace hmm
}; // namespace mlpack

// Include implementation.
#include "hmm_util_impl.hpp"

#endif
