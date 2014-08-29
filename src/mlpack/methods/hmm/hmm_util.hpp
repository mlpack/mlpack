/**
 * @file hmm_util.hpp
 * @author Ryan Curtin
 *
 * Save/load utilities for HMMs.  This should be eventually merged into the HMM
 * class itself.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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
