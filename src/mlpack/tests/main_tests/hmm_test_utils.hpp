/**
 * @file tests/main_tests/hmm_test_utils.hpp
 * @author Daivik Nema
 *
 * Structs for initializing and training HMMs (either of Discrete, Gaussian,
 * GMM, or Diagonal GMM HMMs). These structs are passed as template parameters
 * to the PerformAction function of an HMMModel object. These structs have been
 * adapted from the structs in mlpack/methods/hmm/hmm_train_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_MAIN_TESTS_HMM_TEST_UTILS_HPP
#define MLPACK_TESTS_MAIN_TESTS_HMM_TEST_UTILS_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/hmm/hmm.hpp>

struct InitHMMModel
{
  template<typename HMMType>
  static void Apply(util::Params& /* params */,
                    HMMType& hmm,
                    vector<mat>* trainSeq)
  {
    const size_t states = 2;

    // Create the initialized-to-zero model.
    Create(hmm, *trainSeq, states);

    // Initializing the emission distribution depends on the distribution.
    // Therefore we have to use the helper functions.
    RandomInitialize(hmm.Emission());
  }

  //! Helper function to create discrete HMM.
  static void Create(HMM<DiscreteDistribution>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance = 1e-05)
  {
    // Maximum observation is necessary so we know how to train the discrete
    // distribution.
    arma::Col<size_t> maxEmissions(trainSeq[0].n_rows);
    maxEmissions.zeros();
    for (vector<mat>::iterator it = trainSeq.begin(); it != trainSeq.end();
         ++it)
    {
      arma::Col<size_t> maxSeqs =
          arma::conv_to<arma::Col<size_t>>::from(arma::max(*it, 1)) + 1;
      maxEmissions = arma::max(maxEmissions, maxSeqs);
    }

    hmm = HMM<DiscreteDistribution>(size_t(states),
        DiscreteDistribution(maxEmissions), tolerance);
  }

  static void Create(HMM<GaussianDistribution>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance = 1e-05)
  {
    // Find dimension of the data.
    const size_t dimensionality = trainSeq[0].n_rows;

    // Verify dimensionality of data.
    for (size_t i = 0; i < trainSeq.size(); ++i)
    {
      if (trainSeq[i].n_rows != dimensionality)
      {
        Log::Fatal << "Observation sequence " << i << " dimensionality ("
            << trainSeq[i].n_rows << " is incorrect (should be "
            << dimensionality << ")!" << endl;
      }
    }

    // Get the model and initialize it.
    hmm = HMM<GaussianDistribution>(size_t(states),
        GaussianDistribution(dimensionality), tolerance);
  }

  static void Create(HMM<GMM>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance = 1e-05)
  {
    // Find dimension of the data.
    const size_t dimensionality = trainSeq[0].n_rows;
    const int gaussians = 2;

    if (gaussians == 0)
    {
      Log::Fatal << "Number of gaussians for each GMM must be specified "
          << "when type = 'gmm'!" << endl;
    }

    if (gaussians < 0)
    {
      Log::Fatal << "Invalid number of gaussians (" << gaussians << "); must "
          << "be greater than or equal to 1." << endl;
    }

    // Create HMM object.
    hmm = HMM<GMM>(size_t(states), GMM(size_t(gaussians), dimensionality),
        tolerance);
  }

  //! Helper function to create Diagonal GMM HMM.
  static void Create(HMM<DiagonalGMM>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance = 1e-05)
  {
    // Find dimension of the data.
    const size_t dimensionality = trainSeq[0].n_rows;
    const int gaussians = 2;

    if (gaussians == 0)
    {
      Log::Fatal << "Number of gaussians for each GMM must be specified "
          << "when type = 'diag_gmm'!" << endl;
    }

    if (gaussians < 0)
    {
      Log::Fatal << "Invalid number of gaussians (" << gaussians << "); must "
          << "be greater than or equal to 1." << endl;
    }

    // Create HMM object.
    hmm = HMM<DiagonalGMM>(size_t(states), DiagonalGMM(size_t(gaussians),
        dimensionality), tolerance);
  }

  //! Helper function for discrete emission distributions.
  static void RandomInitialize(vector<DiscreteDistribution>& e)
  {
    for (size_t i = 0; i < e.size(); ++i)
    {
      e[i].Probabilities().randu();
      e[i].Probabilities() /= arma::accu(e[i].Probabilities());
    }
  }

  static void RandomInitialize(vector<GaussianDistribution>& e)
  {
    for (size_t i = 0; i < e.size(); ++i)
    {
      const size_t dimensionality = e[i].Mean().n_rows;
      e[i].Mean().randu();
      // Generate random covariance.
      arma::mat r = arma::randu<arma::mat>(dimensionality, dimensionality);
      e[i].Covariance(r * r.t());
    }
  }

  static void RandomInitialize(vector<GMM>& e)
  {
    for (size_t i = 0; i < e.size(); ++i)
    {
      // Random weights.
      e[i].Weights().randu();
      e[i].Weights() /= arma::accu(e[i].Weights());

      // Random means and covariances.
      for (int g = 0; g < 2; ++g)
      {
        const size_t dimensionality = e[i].Component(g).Mean().n_rows;
        e[i].Component(g).Mean().randu();

        // Generate random covariance.
        arma::mat r = arma::randu<arma::mat>(dimensionality,
            dimensionality);
        e[i].Component(g).Covariance(r * r.t());
      }
    }
  }

  //! Helper function for diagonal GMM emission distributions.
  static void RandomInitialize(vector<DiagonalGMM>& e)
  {
    for (size_t i = 0; i < e.size(); ++i)
    {
      // Random weights.
      e[i].Weights().randu();
      e[i].Weights() /= arma::accu(e[i].Weights());

      // Random means and covariances.
      for (int g = 0; g < 2; ++g)
      {
        const size_t dimensionality = e[i].Component(g).Mean().n_rows;
        e[i].Component(g).Mean().randu();

        // Generate random diagonal covariance.
        arma::vec r = arma::randu<arma::vec>(dimensionality);
        e[i].Component(g).Covariance(r);
      }
    }
  }
};

struct TrainHMMModel
{
  template<typename HMMType>
  static void Apply(util::Params& /* params */,
                    HMMType& hmm,
                    vector<arma::mat>* trainSeq)
  {
    // For now, perform unsupervised (Baum-Welch) training.
    hmm.Train(*trainSeq);
  }
};

#endif
