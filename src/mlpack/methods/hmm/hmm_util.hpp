/**
 * @file hmm_util.hpp
 * @author Ryan Curtin
 *
 * Utility to read HMM type from file.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HMM_HMM_UTIL_HPP
#define MLPACK_METHODS_HMM_HMM_UTIL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace hmm {

//! HMMType, to be stored on disk.  This is of type char, which is one byte.
//! (I'm not sure what will happen on systems where one byte is not eight bits.)
enum HMMType : char
{
  DiscreteHMM = 0,
  GaussianHMM,
  GaussianMixtureModelHMM
};

//! ActionType should implement static void Apply(HMMType&).
template<typename ActionType, typename ExtraInfoType = void>
void LoadHMMAndPerformAction(const std::string& modelFile,
                             ExtraInfoType* x = NULL);

//! Save an HMM to a file.  The file must also encode what type of HMM is being
//! stored.
template<typename HMMType>
void SaveHMM(HMMType& hmm, const std::string& modelFile);

} // namespace hmm
} // namespace mlpack

#include "hmm_util_impl.hpp"

#endif
