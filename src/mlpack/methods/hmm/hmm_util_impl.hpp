/**
 * @file hmm_util_impl.hpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Implementation of HMM load/save functions.
 */
#ifndef __MLPACK_METHODS_HMM_HMM_UTIL_IMPL_HPP
#define __MLPACK_METHODS_HMM_HMM_UTIL_IMPL_HPP

// In case it hasn't already been included.
#include "hmm_util.hpp"
// Only required for conversion util
#include <mlpack/methods/gmm/gmm.hpp>

namespace mlpack {
namespace hmm {

/**
 * Save an HMM to file (deprecated).
 *
 * @tparam Distribution Distribution type of HMM.
 * @param sr SaveRestoreUtility to use.
 */
template<typename Distribution>
void SaveHMM(const HMM<Distribution>& hmm, util::SaveRestoreUtility& sr)
{
  Log::Warn << "SaveHMM is deprecated. See HMM::Save.";
  hmm.Save(sr);
}

/**
 * Load an HMM from file (deprecated).
 *
 * @tparam Distribution Distribution type of HMM.
 * @param sr SaveRestoreUtility to use.
 */
template<typename Distribution>
void LoadHMM(HMM<Distribution>& hmm, util::SaveRestoreUtility& sr)
{
  Log::Warn << "LoadHMM is deprecated. See HMM::Load.";
  hmm.Load(sr);
}

/**
 * Converter for HMMs saved using older MLPACK versions.
 *
 * @tparam Distribution Distribution type of HMM.
 * @param sr SaveRestoreUtility to use.
 */
template<typename Distribution>
void ConvertHMM(HMM<Distribution>& /* hmm */,
                const util::SaveRestoreUtility& /* sr */)
{
  Log::Fatal << "HMM conversion not implemented for arbitrary distributions."
      << std::endl;
}

template<>
void ConvertHMM(HMM<distribution::DiscreteDistribution>& hmm,
                const util::SaveRestoreUtility& sr)
{
  std::string type;
  size_t states;

  sr.LoadParameter(type, "hmm_type");
  if (type != "discrete")
  {
    Log::Fatal << "Cannot load non-discrete HMM (of type " << type << ") as "
        << "discrete HMM!" << std::endl;
  }

  sr.LoadParameter(states, "hmm_states");

  // Load transition matrix.
  sr.LoadParameter(hmm.Transition(), "hmm_transition");

  // Now each emission distribution.
  hmm.Emission().resize(states);
  for (size_t i = 0; i < states; ++i)
  {
    std::stringstream s;
    s << "hmm_emission_distribution_" << i;
    sr.LoadParameter(hmm.Emission()[i].Probabilities(), s.str());
  }

  hmm.Dimensionality() = 1;
}

template<>
void ConvertHMM(HMM<distribution::GaussianDistribution>& hmm,
                const util::SaveRestoreUtility& sr)
{
  std::string type;
  size_t states;

  sr.LoadParameter(type, "hmm_type");
  if (type != "gaussian")
  {
    Log::Fatal << "Cannot load non-Gaussian HMM (of type " << type << ") as "
        << "a Gaussian HMM!" << std::endl;
  }

  sr.LoadParameter(states, "hmm_states");

  // Load transition matrix.
  sr.LoadParameter(hmm.Transition(), "hmm_transition");

  // Now each emission distribution.
  hmm.Emission().resize(states);
  for (size_t i = 0; i < states; ++i)
  {
    std::stringstream s;
    s << "hmm_emission_mean_" << i;
    sr.LoadParameter(hmm.Emission()[i].Mean(), s.str());

    s.str("");
    s << "hmm_emission_covariance_" << i;
    sr.LoadParameter(hmm.Emission()[i].Covariance(), s.str());
  }

  hmm.Dimensionality() = hmm.Emission()[0].Mean().n_elem;
}

template<>
void ConvertHMM(HMM<gmm::GMM<> >& hmm, const util::SaveRestoreUtility& sr)
{
  std::string type;
  size_t states;

  sr.LoadParameter(type, "hmm_type");
  if (type != "gmm")
  {
    Log::Fatal << "Cannot load non-GMM HMM (of type " << type << ") as "
        << "a Gaussian Mixture Model HMM!" << std::endl;
  }

  sr.LoadParameter(states, "hmm_states");

  // Load transition matrix.
  sr.LoadParameter(hmm.Transition(), "hmm_transition");

  // Now each emission distribution.
  hmm.Emission().resize(states, gmm::GMM<>(1, 1));
  for (size_t i = 0; i < states; ++i)
  {
    std::stringstream s;
    s << "hmm_emission_" << i << "_gaussians";
    size_t gaussians;
    sr.LoadParameter(gaussians, s.str());

    s.str("");
    // Extract dimensionality.
    arma::vec meanzero;
    s << "hmm_emission_" << i << "_gaussian_0_mean";
    sr.LoadParameter(meanzero, s.str());
    size_t dimensionality = meanzero.n_elem;

    // Initialize GMM correctly.
    hmm.Emission()[i].Gaussians() = gaussians;
    hmm.Emission()[i].Dimensionality() = dimensionality;

    for (size_t g = 0; g < gaussians; ++g)
    {
      s.str("");
      s << "hmm_emission_" << i << "_gaussian_" << g << "_mean";
      sr.LoadParameter(hmm.Emission()[i].Component(g).Mean(), s.str());

      s.str("");
      s << "hmm_emission_" << i << "_gaussian_" << g << "_covariance";
      sr.LoadParameter(hmm.Emission()[i].Component(g).Covariance(), s.str());
    }

    s.str("");
    s << "hmm_emission_" << i << "_weights";
    sr.LoadParameter(hmm.Emission()[i].Weights(), s.str());
  }

  hmm.Dimensionality() = hmm.Emission()[0].Dimensionality();
}

}; // namespace hmm
}; // namespace mlpack

#endif
