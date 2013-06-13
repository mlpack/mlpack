/**
 * @file hmm_util_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of HMM load/save functions.
 *
 * This file is part of MLPACK 1.0.6.
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
#ifndef __MLPACK_METHODS_HMM_HMM_UTIL_IMPL_HPP
#define __MLPACK_METHODS_HMM_HMM_UTIL_IMPL_HPP

// In case it hasn't already been included.
#include "hmm_util.hpp"

#include <mlpack/methods/gmm/gmm.hpp>

namespace mlpack {
namespace hmm {

template<typename Distribution>
void SaveHMM(const HMM<Distribution>& hmm, util::SaveRestoreUtility& sr)
{
  Log::Fatal << "HMM save not implemented for arbitrary distributions."
      << std::endl;
}

template<>
void SaveHMM(const HMM<distribution::DiscreteDistribution>& hmm,
             util::SaveRestoreUtility& sr)
{
  std::string type = "discrete";
  size_t states = hmm.Transition().n_rows;

  sr.SaveParameter(type, "hmm_type");
  sr.SaveParameter(states, "hmm_states");
  sr.SaveParameter(hmm.Transition(), "hmm_transition");

  // Now the emissions.
  for (size_t i = 0; i < states; ++i)
  {
    // Generate name.
    std::stringstream s;
    s << "hmm_emission_distribution_" << i;
    sr.SaveParameter(hmm.Emission()[i].Probabilities(), s.str());
  }
}

template<>
void SaveHMM(const HMM<distribution::GaussianDistribution>& hmm,
             util::SaveRestoreUtility& sr)
{
  std::string type = "gaussian";
  size_t states = hmm.Transition().n_rows;

  sr.SaveParameter(type, "hmm_type");
  sr.SaveParameter(states, "hmm_states");
  sr.SaveParameter(hmm.Transition(), "hmm_transition");

  // Now the emissions.
  for (size_t i = 0; i < states; ++i)
  {
    // Generate name.
    std::stringstream s;
    s << "hmm_emission_mean_" << i;
    sr.SaveParameter(hmm.Emission()[i].Mean(), s.str());

    s.str("");
    s << "hmm_emission_covariance_" << i;
    sr.SaveParameter(hmm.Emission()[i].Covariance(), s.str());
  }
}

template<>
void SaveHMM(const HMM<gmm::GMM<> >& hmm,
             util::SaveRestoreUtility& sr)
{
  std::string type = "gmm";
  size_t states = hmm.Transition().n_rows;

  sr.SaveParameter(type, "hmm_type");
  sr.SaveParameter(states, "hmm_states");
  sr.SaveParameter(hmm.Transition(), "hmm_transition");

  // Now the emissions.
  for (size_t i = 0; i < states; ++i)
  {
    // Generate name.
    std::stringstream s;
    s << "hmm_emission_" << i << "_gaussians";
    sr.SaveParameter(hmm.Emission()[i].Gaussians(), s.str());

    s.str("");
    s << "hmm_emission_" << i << "_weights";
    sr.SaveParameter(hmm.Emission()[i].Weights(), s.str());

    for (size_t g = 0; g < hmm.Emission()[i].Gaussians(); ++g)
    {
      s.str("");
      s << "hmm_emission_" << i << "_gaussian_" << g << "_mean";
      sr.SaveParameter(hmm.Emission()[i].Means()[g], s.str());

      s.str("");
      s << "hmm_emission_" << i << "_gaussian_" << g << "_covariance";
      sr.SaveParameter(hmm.Emission()[i].Covariances()[g], s.str());
    }
  }
}

template<typename Distribution>
void LoadHMM(HMM<Distribution>& hmm, util::SaveRestoreUtility& sr)
{
  Log::Fatal << "HMM load not implemented for arbitrary distributions."
      << std::endl;
}

template<>
void LoadHMM(HMM<distribution::DiscreteDistribution>& hmm,
             util::SaveRestoreUtility& sr)
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
void LoadHMM(HMM<distribution::GaussianDistribution>& hmm,
             util::SaveRestoreUtility& sr)
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
void LoadHMM(HMM<gmm::GMM<> >& hmm,
             util::SaveRestoreUtility& sr)
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
      sr.LoadParameter(hmm.Emission()[i].Means()[g], s.str());

      s.str("");
      s << "hmm_emission_" << i << "_gaussian_" << g << "_covariance";
      sr.LoadParameter(hmm.Emission()[i].Covariances()[g], s.str());
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
