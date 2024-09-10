/**
 * @file methods/hmm/hmm_util_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of HMM utilities to load arbitrary HMM types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HMM_HMM_UTIL_IMPL_HPP
#define MLPACK_METHODS_HMM_HMM_UTIL_IMPL_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/methods/gmm/diagonal_gmm.hpp>

namespace mlpack {

// Forward declarations of utility functions.

// Set up the archive for deserialization.
template<typename ActionType, typename ArchiveType, typename ExtraInfoType>
void LoadHMMAndPerformActionHelper(const std::string& modelFile,
                                   ExtraInfoType* x = NULL);

// Actually deserialize into the correct type.
template<typename ActionType,
         typename ArchiveType,
         typename HMMType,
         typename ExtraInfoType>
void DeserializeHMMAndPerformAction(ArchiveType& ar, ExtraInfoType* x = NULL);

template<typename ActionType, typename ExtraInfoType>
void LoadHMMAndPerformAction(const std::string& modelFile,
                             ExtraInfoType* x)
{
  const std::string extension = data::Extension(modelFile);
  if (extension == "xml")
  {
    LoadHMMAndPerformActionHelper<ActionType, cereal::XMLInputArchive>(
        modelFile, x);
  }
  else if (extension == "bin")
  {
    LoadHMMAndPerformActionHelper<ActionType, cereal::BinaryInputArchive>(
        modelFile, x);
  }
  else if (extension == "json")
  {
    LoadHMMAndPerformActionHelper<ActionType, cereal::JSONInputArchive>(
        modelFile, x);
  }
  else
  {
    Log::Fatal << "Unknown extension '" << extension << "' for HMM model file "
        << "(known: 'xml', 'json', 'bin')." << std::endl;
  }
}

template<typename ActionType,
         typename ArchiveType,
         typename ExtraInfoType>
void LoadHMMAndPerformActionHelper(const std::string& modelFile,
                                   ExtraInfoType* x)
{
  std::ifstream ifs(modelFile);
  if (ifs.fail())
    Log::Fatal << "Cannot open model file '" << modelFile << "' for loading!"
        << std::endl;
  ArchiveType ar(ifs);

  // Read in the unsigned integer that denotes the type of the model.
  char type;
  ar(CEREAL_NVP(type));

  switch (type)
  {
    case HMMType::DiscreteHMM:
      DeserializeHMMAndPerformAction<ActionType, ArchiveType,
          HMM<DiscreteDistribution<>>>(ar, x);
      break;

    case HMMType::GaussianHMM:
      DeserializeHMMAndPerformAction<ActionType, ArchiveType,
          HMM<GaussianDistribution<>>>(ar, x);
      break;

    case HMMType::GaussianMixtureModelHMM:
      DeserializeHMMAndPerformAction<ActionType, ArchiveType,
          HMM<GMM>>(ar, x);
      break;

    case HMMType::DiagonalGaussianMixtureModelHMM:
      DeserializeHMMAndPerformAction<ActionType, ArchiveType,
          HMM<DiagonalGMM>>(ar, x);

    default:
      Log::Fatal << "Unknown HMM type '" << (unsigned int) type << "'!"
          << std::endl;
  }
}

template<typename ActionType,
         typename ArchiveType,
         typename HMMType,
         typename ExtraInfoType>
void DeserializeHMMAndPerformAction(ArchiveType& ar, ExtraInfoType* x)
{
  // Extract the HMM and perform the action.
  HMMType hmm;
  ar(CEREAL_NVP(hmm));
  ActionType::Apply(hmm, x);
}

// Helper function.
template<typename ArchiveType, typename HMMType>
void SaveHMMHelper(HMMType& hmm, const std::string& modelFile);

template<typename HMMType>
char GetHMMType();

template<typename HMMType>
void SaveHMM(HMMType& hmm, const std::string& modelFile)
{
  const std::string extension = data::Extension(modelFile);
  if (extension == "xml")
    SaveHMMHelper<cereal::XMLOutputArchive>(hmm, modelFile);
  else if (extension == "bin")
    SaveHMMHelper<cereal::BinaryOutputArchive>(hmm, modelFile);
  else if (extension == "json")
    SaveHMMHelper<cereal::JSONOutputArchive>(hmm, modelFile);
  else
    Log::Fatal << "Unknown extension '" << extension << "' for HMM model file."
        << std::endl;
}

template<typename ArchiveType, typename HMMType>
void SaveHMMHelper(HMMType& hmm, const std::string& modelFile)
{
  std::ofstream ofs(modelFile);
  if (ofs.fail())
    Log::Fatal << "Cannot open model file '" << modelFile << "' for saving!"
        << std::endl;
  ArchiveType ar(ofs);

  // Write out the unsigned integer that denotes the type of the model.
  char type = GetHMMType<HMMType>();
  if (type == char(-1))
    Log::Fatal << "Unknown HMM type given to SaveHMM()!" << std::endl;

  ar(CEREAL_NVP(type));
  ar(CEREAL_NVP(hmm));
}

// Utility functions to turn a type into something we can store.
template<typename HMMType>
char GetHMMType() { return char(-1); }

template<>
char GetHMMType<HMM<DiscreteDistribution<>>>()
{
  return HMMType::DiscreteHMM;
}

template<>
char GetHMMType<HMM<GaussianDistribution<>>>()
{
  return HMMType::GaussianHMM;
}

template<>
char GetHMMType<HMM<GMM>>()
{
  return HMMType::GaussianMixtureModelHMM;
}

template<>
char GetHMMType<HMM<DiagonalGMM>>()
{
  return HMMType::DiagonalGaussianMixtureModelHMM;
}

} // namespace mlpack

#endif
