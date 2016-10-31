/**
 * @file hmm_util_impl.hpp
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

#include <mlpack/core.hpp>

#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/gmm/gmm.hpp>

namespace mlpack {
namespace hmm {

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
  using namespace boost::archive;

  const std::string extension = data::Extension(modelFile);
  if (extension == "xml")
    LoadHMMAndPerformActionHelper<ActionType, xml_iarchive>(modelFile, x);
  else if (extension == "bin")
    LoadHMMAndPerformActionHelper<ActionType, binary_iarchive>(modelFile, x);
  else if (extension == "txt")
    LoadHMMAndPerformActionHelper<ActionType, text_iarchive>(modelFile, x);
  else
    Log::Fatal << "Unknown extension '" << extension << "' for HMM model file "
        << "(known: 'xml', 'txt', 'bin')." << std::endl;
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
  ar >> data::CreateNVP(type, "hmm_type");

  using namespace mlpack::distribution;

  switch (type)
  {
    case HMMType::DiscreteHMM:
      DeserializeHMMAndPerformAction<ActionType, ArchiveType,
          HMM<DiscreteDistribution>>(ar, x);
      break;

    case HMMType::GaussianHMM:
      DeserializeHMMAndPerformAction<ActionType, ArchiveType,
          HMM<GaussianDistribution>>(ar, x);
      break;

    case HMMType::GaussianMixtureModelHMM:
      DeserializeHMMAndPerformAction<ActionType, ArchiveType,
          HMM<gmm::GMM>>(ar, x);
      break;

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
  ar >> data::CreateNVP(hmm, "hmm");
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
  using namespace boost::archive;

  const std::string extension = data::Extension(modelFile);
  if (extension == "xml")
    SaveHMMHelper<xml_oarchive>(hmm, modelFile);
  else if (extension == "bin")
    SaveHMMHelper<binary_oarchive>(hmm, modelFile);
  else if (extension == "txt")
    SaveHMMHelper<text_oarchive>(hmm, modelFile);
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

  ar << data::CreateNVP(type, "hmm_type");
  ar << data::CreateNVP(hmm, "hmm");
}

// Utility functions to turn a type into something we can store.
template<typename HMMType>
char GetHMMType() { return char(-1); }

template<>
char GetHMMType<HMM<distribution::DiscreteDistribution>>()
{
  return HMMType::DiscreteHMM;
}

template<>
char GetHMMType<HMM<distribution::GaussianDistribution>>()
{
  return HMMType::GaussianHMM;
}

template<>
char GetHMMType<HMM<gmm::GMM>>()
{
  return HMMType::GaussianMixtureModelHMM;
}

} // namespace hmm
} // namespace mlpack

#endif
