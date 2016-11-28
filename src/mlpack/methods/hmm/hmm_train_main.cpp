/**
 * @file hmm_train_main.cpp
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

#include "hmm.hpp"
#include "hmm_model.hpp"

#include <mlpack/methods/gmm/gmm.hpp>

using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;
using namespace mlpack::util;
using namespace mlpack::gmm;
using namespace mlpack::math;
using namespace arma;
using namespace std;

PROGRAM_INFO("Hidden Markov Model (HMM) Training", "This program allows a "
    "Hidden Markov Model to be trained on labeled or unlabeled data.  It "
    "support three types of HMMs: discrete HMMs, Gaussian HMMs, or GMM HMMs."
    "\n\n"
    "Either one input sequence can be specified (with --input_file), or, a "
    "file containing files in which input sequences can be found (when "
    "--input_file and --batch are used together).  In addition, labels can be "
    "provided in the file specified by --labels_file, and if --batch is used, "
    "the file given to --labels_file should contain a list of files of labels "
    "corresponding to the sequences in the file given to --input_file."
    "\n\n"
    "The HMM is trained with the Baum-Welch algorithm if no labels are "
    "provided.  The tolerance of the Baum-Welch algorithm can be set with the "
    "--tolerance option.  In general it is a good idea to use random "
    "initialization in this case, which can be specified with the "
    "--random_initialization (-r) option."
    "\n\n"
    "Optionally, a pre-created HMM model can be used as a guess for the "
    "transition matrix and emission probabilities; this is specifiable with "
    "--model_file.");

PARAM_STRING_IN_REQ("input_file", "File containing input observations.", "i");
PARAM_STRING_IN_REQ("type", "Type of HMM: discrete | gaussian | gmm.", "t");

PARAM_FLAG("batch", "If true, input_file (and if passed, labels_file) are "
    "expected to contain a list of files to use as input observation sequences "
    "(and label sequences).", "b");
PARAM_INT_IN("states", "Number of hidden states in HMM (necessary, unless "
    "model_file is specified.", "n", 0);
PARAM_INT_IN("gaussians", "Number of gaussians in each GMM (necessary when type"
    " is 'gmm'.", "g", 0);
PARAM_MODEL_IN(HMMModel, "input_model", "Pre-existing HMM model to initialize "
    "training with.", "m");
PARAM_STRING_IN("labels_file", "Optional file of hidden states, used for "
    "labeled training.", "l", "");
PARAM_MODEL_OUT(HMMModel, "output_model", "Output for trained HMM.", "M");
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_DOUBLE_IN("tolerance", "Tolerance of the Baum-Welch algorithm.", "T",
    1e-5);
PARAM_FLAG("random_initialization", "Initialize emissions and transition "
    "matrices with a uniform random distribution.", "r");

// Because we don't know what the type of our HMM is, we need to write a
// function that can take arbitrary HMM types.
struct Init
{
  template<typename HMMType>
  static void Apply(HMMType& hmm, vector<mat>* trainSeq)
  {
    const size_t states = CLI::GetParam<int>("states");
    const double tolerance = CLI::GetParam<double>("tolerance");

    // Create the initialized-to-zero model.
    Create(hmm, *trainSeq, states, tolerance);

    // Initialize the transition matrix and emission distributions, if needed.
    if (CLI::HasParam("random_initialization"))
    {
      hmm.Transition().randu();
      for (size_t c = 0; c < hmm.Transition().n_cols; ++c)
        hmm.Transition().col(c) /= arma::accu(hmm.Transition().col(c));

      // Initializing the emission distribution depends on the distribution.
      // Therefore we have to use the helper functions.
      RandomInitialize(hmm.Emission());
    }
  }

  //! Helper function to create discrete HMM.
  static void Create(HMM<DiscreteDistribution>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance)
  {
    // Maximum observation is necessary so we know how to train the discrete
    // distribution.
    size_t maxEmission = 0;
    for (vector<mat>::iterator it = trainSeq.begin(); it != trainSeq.end();
         ++it)
    {
      size_t maxSeq = size_t(as_scalar(max(trainSeq[0], 1))) + 1;
      if (maxSeq > maxEmission)
        maxEmission = maxSeq;
    }

    Log::Info << maxEmission << " discrete observations in the input data."
        << endl;

    hmm = HMM<DiscreteDistribution>(size_t(states),
        DiscreteDistribution(maxEmission), tolerance);
  }

  //! Helper function to create Gaussian HMM.
  static void Create(HMM<GaussianDistribution>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance)
  {
    // Find dimension of the data.
    const size_t dimensionality = trainSeq[0].n_rows;

    // Verify dimensionality of data.
    for (size_t i = 0; i < trainSeq.size(); ++i)
      if (trainSeq[i].n_rows != dimensionality)
        Log::Fatal << "Observation sequence " << i << " dimensionality ("
            << trainSeq[i].n_rows << " is incorrect (should be "
            << dimensionality << ")!" << endl;

    // Get the model and initialize it.
    hmm = HMM<GaussianDistribution>(size_t(states),
        GaussianDistribution(dimensionality), tolerance);
  }

  //! Helper function to create GMM HMM.
  static void Create(HMM<GMM>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance)
  {
    // Find dimension of the data.
    const size_t dimensionality = trainSeq[0].n_rows;
    const int gaussians = CLI::GetParam<int>("gaussians");

    if (gaussians == 0)
      Log::Fatal << "Number of gaussians for each GMM must be specified (-g) "
          << "when type = 'gmm'!" << endl;

    if (gaussians < 0)
      Log::Fatal << "Invalid number of gaussians (" << gaussians << "); must "
          << "be greater than or equal to 1." << endl;

    // Create HMM object.
    hmm = HMM<GMM>(size_t(states), GMM(size_t(gaussians), dimensionality),
        tolerance);

    // Issue a warning if the user didn't give labels.
    if (!CLI::HasParam("labels_file"))
      Log::Warn << "Unlabeled training of GMM HMMs is almost certainly not "
          << "going to produce good results!" << endl;
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

  //! Helper function for Gaussian emission distributions.
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

  //! Helper function for GMM emission distributions.
  static void RandomInitialize(vector<GMM>& e)
  {
    for (size_t i = 0; i < e.size(); ++i)
    {
      // Random weights.
      e[i].Weights().randu();
      e[i].Weights() /= arma::accu(e[i].Weights());

      // Random means and covariances.
      for (int g = 0; g < CLI::GetParam<int>("gaussians"); ++g)
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
};

// Because we don't know what the type of our HMM is, we need to write a
// function that can take arbitrary HMM types.
struct Train
{
  template<typename HMMType>
  static void Apply(HMMType& hmm, vector<mat>* trainSeqPtr)
  {
    const bool batch = CLI::HasParam("batch");
    const double tolerance = CLI::GetParam<double>("tolerance");

    // Do we need to replace the tolerance?
    if (CLI::HasParam("tolerance"))
      hmm.Tolerance() = tolerance;

    const string labelsFile = CLI::GetParam<string>("labels_file");

    // Verify that the dimensionality of our observations is the same as the
    // dimensionality of our HMM's emissions.
    vector<mat>& trainSeq = *trainSeqPtr;
    for (size_t i = 0; i < trainSeq.size(); ++i)
      if (trainSeq[i].n_rows != hmm.Emission()[0].Dimensionality())
        Log::Fatal << "Dimensionality of training sequence " << i << " ("
            << trainSeq[i].n_rows << ") is not equal to the dimensionality of "
            << "the HMM (" << hmm.Emission()[0].Dimensionality() << ")!"
            << endl;

    vector<arma::Row<size_t>> labelSeq; // May be empty.
    if (CLI::HasParam("labels_file"))
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
          Log::Fatal << "Label sequence " << labelSeq.size() << " does not have"
              << " the same number of points as observation sequence "
              << labelSeq.size() << "!" << endl;

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

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  // Set random seed.
  if (CLI::GetParam<int>("seed") != 0)
    RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    RandomSeed((size_t) time(NULL));

  // Validate parameters.
  const string inputFile = CLI::GetParam<string>("input_file");
  const string type = CLI::GetParam<string>("type");
  const bool batch = CLI::HasParam("batch");
  const double tolerance = CLI::GetParam<double>("tolerance");

  // Verify that either a model or a type was given.
  if (!CLI::HasParam("input_model") && type == "")
    Log::Fatal << "No model file specified and no HMM type given!  At least "
        << "one is required." << endl;

  // If no model is specified, make sure we are training with valid parameters.
  if (!CLI::HasParam("input_model"))
  {
    // Validate number of states.
    if (!CLI::HasParam("states"))
      Log::Fatal << "Must specify number of states if model file is not "
          << "specified!" << endl;
    if (CLI::GetParam<int>("states") <= 0)
      Log::Fatal << "Must specify a positive number of states!" << endl;
  }

  if (CLI::HasParam("input_model") && CLI::HasParam("tolerance"))
    Log::Info << "Tolerance of existing model in '"
        << CLI::GetUnmappedParam<std::string>("input_model") << "' will be "
        << "replaced with specified tolerance of " << tolerance << "." << endl;

  if (CLI::HasParam("input_model") && CLI::HasParam("type"))
    Log::Warn << "--type ignored because --input_model_file specified." << endl;

  if (!CLI::HasParam("input_model") &&
      (type != "discrete") && (type != "gaussian") && (type != "gmm"))
    Log::Fatal << "Unknown type '" << type << "'; must be 'discrete', "
        << "'gaussian', or 'gmm'!" << endl;

  // Load the input data.
  vector<mat> trainSeq;
  if (batch)
  {
    // The input file contains a list of files to read.
    Log::Info << "Reading list of training sequences from '" << inputFile
        << "'." << endl;

    fstream f(inputFile.c_str(), ios_base::in);

    if (!f.is_open())
      Log::Fatal << "Could not open '" << inputFile << "' for reading."
          << endl;

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

  // If we have a model file, we can autodetect the type.
  HMMModel hmm(typeId);
  if (CLI::HasParam("input_model"))
  {
    hmm = std::move(CLI::GetParam<HMMModel>("input_model"));
  }
  else
  {
    // We need to initialize the model.
    hmm.PerformAction<Init, vector<mat>>(&trainSeq);
  }

  // Train the model.
  hmm.PerformAction<Train, vector<mat>>(&trainSeq);

  // If necessary, save the output.
  if (CLI::HasParam("output_model"))
    CLI::GetParam<HMMModel>("output_model") = std::move(hmm);

  CLI::Destroy();
}
