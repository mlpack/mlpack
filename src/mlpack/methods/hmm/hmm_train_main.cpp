/**
 * @file methods/hmm/hmm_train_main.cpp
 * @author Ryan Curtin
 *
 * Executable which trains an HMM and saves the trained HMM to file.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME hmm_train

#include <mlpack/core/util/mlpack_main.hpp>

#include "hmm.hpp"
#include "hmm_model.hpp"

#include <mlpack/methods/gmm/gmm.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_USER_NAME("Hidden Markov Model (HMM) Training");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of training algorithms for Hidden Markov Models (HMMs). "
    "Given labeled or unlabeled data, an HMM can be trained for further use "
    "with other mlpack HMM tools.");

// Long description.
BINDING_LONG_DESC(
    "This program allows a Hidden Markov Model to be trained on labeled or "
    "unlabeled data.  It supports four types of HMMs: Discrete HMMs, "
    "Gaussian HMMs, GMM HMMs, or Diagonal GMM HMMs"
    "\n\n"
    "Either one input sequence can be specified (with " +
    PRINT_PARAM_STRING("input_file") + "), or, a file containing files in "
    "which input sequences can be found (when "+
    PRINT_PARAM_STRING("input_file") + "and" + PRINT_PARAM_STRING("batch") +
    " are used together).  In addition, labels can be "
    "provided in the file specified by " + PRINT_PARAM_STRING("labels_file") +
    ", and if " + PRINT_PARAM_STRING("batch") + " is used, "
    "the file given to " + PRINT_PARAM_STRING("labels_file") +
    " should contain a list of files of labels corresponding to the sequences"
    " in the file given to " + PRINT_PARAM_STRING("input_file") + "."
    "\n\n"
    "The HMM is trained with the Baum-Welch algorithm if no labels are "
    "provided.  The tolerance of the Baum-Welch algorithm can be set with the "
    + PRINT_PARAM_STRING("tolerance") + "option.  By default, the transition "
    "matrix is randomly initialized and the emission distributions are "
    "initialized to fit the extent of the data."
    "\n\n"
    "Optionally, a pre-created HMM model can be used as a guess for the "
    "transition matrix and emission probabilities; this is specifiable with " +
    PRINT_PARAM_STRING("output_model") + ".");

// See also...
BINDING_SEE_ALSO("@hmm_generate", "#hmm_generate");
BINDING_SEE_ALSO("@hmm_loglik", "#hmm_loglik");
BINDING_SEE_ALSO("@hmm_viterbi", "#hmm_viterbi");
BINDING_SEE_ALSO("Hidden Mixture Models on Wikipedia",
    "https://en.wikipedia.org/wiki/Hidden_Markov_model");
BINDING_SEE_ALSO("HMM class documentation", "@src/mlpack/methods/hmm/hmm.hpp");

PARAM_STRING_IN_REQ("input_file", "File containing input observations.", "i");
PARAM_STRING_IN("type", "Type of HMM: discrete | gaussian | diag_gmm | gmm.",
    "t", "gaussian");

PARAM_FLAG("batch", "If true, input_file (and if passed, labels_file) are "
    "expected to contain a list of files to use as input observation sequences "
    "(and label sequences).", "b");
PARAM_INT_IN("states", "Number of hidden states in HMM (necessary, unless "
    "model_file is specified).", "n", 0);
PARAM_INT_IN("gaussians", "Number of gaussians in each GMM (necessary when type"
    " is 'gmm').", "g", 0);
PARAM_MODEL_IN(HMMModel, "input_model", "Pre-existing HMM model to initialize "
    "training with.", "m");
PARAM_STRING_IN("labels_file", "Optional file of hidden states, used for "
    "labeled training.", "l", "");
PARAM_MODEL_OUT(HMMModel, "output_model", "Output for trained HMM.", "M");
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_DOUBLE_IN("tolerance", "Tolerance of the Baum-Welch algorithm.", "T",
    1e-5);

// Because we don't know what the type of our HMM is, we need to write a
// function that can take arbitrary HMM types.
struct Init
{
  template<typename HMMType>
  static void Apply(util::Params& params, HMMType& hmm, vector<mat>* trainSeq)
  {
    const size_t states = params.Get<int>("states");
    const double tolerance = params.Get<double>("tolerance");

    // Create the initialized-to-zero model.
    Create(params, hmm, *trainSeq, states, tolerance);

    // Initializing the emission distribution depends on the distribution.
    // Therefore we have to use the helper functions.
    RandomInitialize(params, hmm.Emission());
  }

  //! Helper function to create discrete HMM.
  static void Create(util::Params& /* params */,
                     HMM<DiscreteDistribution<>>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance)
  {
    // Maximum observation is necessary so we know how to train the discrete
    // distribution.
    arma::Col<size_t> maxEmissions(trainSeq[0].n_rows);
    maxEmissions.zeros();
    for (vector<mat>::iterator it = trainSeq.begin(); it != trainSeq.end();
         ++it)
    {
      arma::Col<size_t> maxSeqs =
          ConvTo<arma::Col<size_t>>::From(arma::max(*it, 1)) + 1;
      maxEmissions = arma::max(maxEmissions, maxSeqs);
    }

    hmm = HMM<DiscreteDistribution<>>(size_t(states),
        DiscreteDistribution<>(maxEmissions), tolerance);
  }

  //! Helper function to create Gaussian HMM.
  static void Create(util::Params& /* params */,
                     HMM<GaussianDistribution<>>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance)
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
    hmm = HMM<GaussianDistribution<>>(size_t(states),
        GaussianDistribution<>(dimensionality), tolerance);
  }

  //! Helper function to create GMM HMM.
  static void Create(util::Params& params,
                     HMM<GMM>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance)
  {
    // Find dimension of the data.
    const size_t dimensionality = trainSeq[0].n_rows;
    const int gaussians = params.Get<int>("gaussians");

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

    // Issue a warning if the user didn't give labels.
    if (!params.Has("labels_file"))
    {
      Log::Warn << "Unlabeled training of GMM HMMs is almost certainly not "
          << "going to produce good results!" << endl;
    }
  }

  //! Helper function to create Diagonal GMM HMM.
  static void Create(util::Params& params,
                     HMM<DiagonalGMM>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance)
  {
    // Find dimension of the data.
    const size_t dimensionality = trainSeq[0].n_rows;
    const int gaussians = params.Get<int>("gaussians");

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

    // Issue a warning if the user didn't give labels.
    if (!params.Has("labels_file"))
    {
      Log::Warn << "Unlabeled training of Diagonal GMM HMMs is almost "
          << "certainly not going to produce good results!" << endl;
    }
  }

  //! Helper function for discrete emission distributions.
  static void RandomInitialize(util::Params& /* params */,
                               vector<DiscreteDistribution<>>& e)
  {
    for (size_t i = 0; i < e.size(); ++i)
    {
      e[i].Probabilities().randu();
      e[i].Probabilities() /= accu(e[i].Probabilities());
    }
  }

  //! Helper function for Gaussian emission distributions.
  static void RandomInitialize(util::Params& /* params */,
                               vector<GaussianDistribution<>>& e)
  {
    for (size_t i = 0; i < e.size(); ++i)
    {
      const size_t dimensionality = e[i].Mean().n_rows;
      e[i].Mean().randu();
      // Generate random covariance.
      arma::mat r;
      r.randu(dimensionality, dimensionality);
      e[i].Covariance(r * r.t());
    }
  }

  //! Helper function for GMM emission distributions.
  static void RandomInitialize(util::Params& params,
                               vector<GMM>& e)
  {
    for (size_t i = 0; i < e.size(); ++i)
    {
      // Random weights.
      e[i].Weights().randu();
      e[i].Weights() /= accu(e[i].Weights());

      // Random means and covariances.
      for (int g = 0; g < params.Get<int>("gaussians"); ++g)
      {
        const size_t dimensionality = e[i].Component(g).Mean().n_rows;
        e[i].Component(g).Mean().randu();

        // Generate random covariance.
        arma::mat r;
        r.randu(dimensionality, dimensionality);
        e[i].Component(g).Covariance(r * r.t());
      }
    }
  }

  //! Helper function for Diagonal GMM emission distributions.
  static void RandomInitialize(util::Params& params,
                               vector<DiagonalGMM>& e)
  {
    for (size_t i = 0; i < e.size(); ++i)
    {
      // Random weights.
      e[i].Weights().randu();
      e[i].Weights() /= accu(e[i].Weights());

      // Random means and covariances.
      for (int g = 0; g < params.Get<int>("gaussians"); ++g)
      {
        const size_t dimensionality = e[i].Component(g).Mean().n_rows;
        e[i].Component(g).Mean().randu();

        // Generate random diagonal covariance.
        arma::vec r;
        r.randu(dimensionality);
        e[i].Component(g).Covariance(r);
      }
    }
  }
};

// Because we don't know what the type of our HMM is, we need to write a
// function that can take arbitrary HMM types.
struct Train
{
  template<typename HMMType>
  static void Apply(util::Params& params,
                    HMMType& hmm,
                    vector<mat>* trainSeqPtr)
  {
    const bool batch = params.Has("batch");
    const double tolerance = params.Get<double>("tolerance");

    // Do we need to replace the tolerance?
    if (params.Has("tolerance"))
      hmm.Tolerance() = tolerance;

    const string labelsFile = params.Get<string>("labels_file");

    // Verify that the dimensionality of our observations is the same as the
    // dimensionality of our HMM's emissions.
    vector<mat>& trainSeq = *trainSeqPtr;
    for (size_t i = 0; i < trainSeq.size(); ++i)
    {
      if (trainSeq[i].n_rows != hmm.Emission()[0].Dimensionality())
      {
        Log::Fatal << "Dimensionality of training sequence " << i << " ("
            << trainSeq[i].n_rows << ") is not equal to the dimensionality of "
            << "the HMM (" << hmm.Emission()[0].Dimensionality() << ")!"
            << endl;
      }
    }

    vector<arma::Row<size_t>> labelSeq; // May be empty.
    if (params.Has("labels_file"))
    {
      // Do we have multiple label files to load?
      char lineBuf[1024];
      if (batch)
      {
        fstream f(labelsFile);

        if (!f.is_open())
          Log::Fatal << "Could not open '" << labelsFile << "' for reading."
              << endl;

        // Now read each line in.
        f.getline(lineBuf, 1024, '\n');
        while (!f.eof())
        {
          Log::Info << "Adding training sequence labels from '" << lineBuf
              << "'." << endl;

          // Now read the matrix.
          Mat<size_t> label;
          data::Load(lineBuf, label, true); // Fatal on failure.

          // Ensure that matrix only has one row.
          if (label.n_cols == 1)
            label = trans(label);

          if (label.n_rows > 1)
            Log::Fatal << "Invalid labels; must be one-dimensional." << endl;

          // Check all of the labels.
          for (size_t i = 0; i < label.n_cols; ++i)
          {
            if (label[i] >= hmm.Transition().n_cols)
            {
              Log::Fatal << "HMM has " << hmm.Transition().n_cols << " hidden "
                  << "states, but label on line " << i << " of '" << lineBuf
                  << "' is " << label[i] << " (should be between 0 and "
                  << (hmm.Transition().n_cols - 1) << ")!" << endl;
            }
          }

          labelSeq.push_back(label.row(0));

          f.getline(lineBuf, 1024, '\n');
        }

        f.close();
      }
      else
      {
        Mat<size_t> label;
        data::Load(labelsFile, label, true);

        // Ensure that matrix only has one row.
        if (label.n_cols == 1)
          label = trans(label);

        if (label.n_rows > 1)
          Log::Fatal << "Invalid labels; must be one-dimensional." << endl;

        // Verify the same number of observations as the data.
        if (label.n_elem != trainSeq[labelSeq.size()].n_cols)
        {
          Log::Fatal << "Label sequence " << labelSeq.size() << " does not have"
              << " the same number of points as observation sequence "
              << labelSeq.size() << "!" << endl;
        }

        // Check all of the labels.
        for (size_t i = 0; i < label.n_cols; ++i)
        {
          if (label[i] >= hmm.Transition().n_cols)
          {
            Log::Fatal << "HMM has " << hmm.Transition().n_cols << " hidden "
                << "states, but label on line " << i << " of '" << labelsFile
                << "' is " << label[i] << " (should be between 0 and "
                << (hmm.Transition().n_cols - 1) << ")!" << endl;
          }
        }

        labelSeq.push_back(label.row(0));
      }

      // Now perform the training with labels.
      hmm.Train(trainSeq, labelSeq);
    }
    else
    {
      // Perform unsupervised training.
      hmm.Train(trainSeq);
    }
  }
};

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  // Set random seed.
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) time(NULL));

  // Validate parameters.
  const string inputFile = params.Get<string>("input_file");
  const string type = params.Get<string>("type");
  const bool batch = params.Has("batch");
  const double tolerance = params.Get<double>("tolerance");

  // If no model is specified, make sure we are training with valid parameters.
  if (!params.Has("input_model"))
  {
    // Validate number of states.
    RequireAtLeastOnePassed(params, { "states" }, true);
    RequireAtLeastOnePassed(params, { "type" }, true);
    RequireParamValue<int>(params, "states", [](int x) { return x > 0; }, true,
        "number of states must be positive");
  }

  if (params.Has("input_model") && params.Has("tolerance"))
  {
    Log::Info << "Tolerance of existing model in '"
        << params.GetPrintable<HMMModel*>("input_model") << "' will be "
        << "replaced with specified tolerance of " << tolerance << "." << endl;
  }

  ReportIgnoredParam(params, {{ "input_model", true }}, "type");

  if (!params.Has("input_model"))
  {
    RequireParamInSet<string>(params, "type", { "discrete", "gaussian", "gmm",
        "diag_gmm" }, true, "unknown HMM type");
  }

  RequireParamValue<double>(params, "tolerance",
      [](double x) { return x >= 0; }, true, "tolerance must be non-negative");

  // Load the input data.
  vector<mat> trainSeq;
  if (batch)
  {
    // The input file contains a list of files to read.
    Log::Info << "Reading list of training sequences from '" << inputFile
        << "'." << endl;

    fstream f(inputFile.c_str(), ios_base::in);

    if (!f.is_open())
    {
      Log::Fatal << "Could not open '" << inputFile << "' for reading."
          << endl;
    }

    // Now read each line in.
    char lineBuf[1024]; // Max 1024 characters... hopefully long enough.
    f.getline(lineBuf, 1024, '\n');
    while (!f.eof())
    {
      Log::Info << "Adding training sequence from '" << lineBuf << "'."
          << endl;

      // Now read the matrix.
      trainSeq.push_back(mat());
      data::Load(lineBuf, trainSeq.back(), true); // Fatal on failure.

      // See if we need to transpose the data.
      if (type == "discrete")
      {
        if (trainSeq.back().n_cols == 1)
          trainSeq.back() = trans(trainSeq.back());
      }

      f.getline(lineBuf, 1024, '\n');
    }

    f.close();
  }
  else
  {
    // Only one input file.
    trainSeq.resize(1);
    data::Load(inputFile, trainSeq[0], true);
  }

  // Get the type.
  HMMType typeId;
  if (type == "discrete")
    typeId = HMMType::DiscreteHMM;
  else if (type == "gaussian")
    typeId = HMMType::GaussianHMM;
  else if (type == "gmm")
    typeId = HMMType::GaussianMixtureModelHMM;
  else
    typeId = HMMType::DiagonalGaussianMixtureModelHMM;

  // If we have a model file, we can autodetect the type.
  HMMModel* hmm;
  if (params.Has("input_model"))
  {
    hmm = params.Get<HMMModel*>("input_model");

    hmm->PerformAction<Train, vector<mat>>(params, &trainSeq);
  }
  else
  {
    // We need to initialize the model.
    hmm = new HMMModel(typeId);

    // Catch any exceptions so that we can clean the model if needed.
    try
    {
      hmm->PerformAction<Init, vector<mat>>(params, &trainSeq);
      hmm->PerformAction<Train, vector<mat>>(params, &trainSeq);
    }
    catch (std::exception& e)
    {
      delete hmm;
      throw;
    }
  }

  // If necessary, save the output.
  params.Get<HMMModel*>("output_model") = hmm;
}
