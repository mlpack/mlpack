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
#include "hmm_util.hpp"

#include <mlpack/methods/gmm/gmm.hpp>

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
PARAM_STRING_IN("model_file", "Pre-existing HMM model file.", "m", "");
PARAM_STRING_IN("labels_file", "Optional file of hidden states, used for "
    "labeled training.", "l", "");
PARAM_STRING_OUT("output_model_file", "File to save trained HMM to.", "M");
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_DOUBLE_IN("tolerance", "Tolerance of the Baum-Welch algorithm.", "T",
    1e-5);
PARAM_FLAG("random_initialization", "Initialize emissions and transition "
    "matrices with a uniform random distribution.", "r");

using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;
using namespace mlpack::util;
using namespace mlpack::gmm;
using namespace mlpack::math;
using namespace arma;
using namespace std;

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

    // Save the model.
    if (CLI::HasParam("output_model_file"))
    {
      const string modelFile = CLI::GetParam<string>("output_model_file");
      SaveHMM(hmm, modelFile);
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
  const string modelFile = CLI::GetParam<string>("model_file");
  const string inputFile = CLI::GetParam<string>("input_file");
  const string type = CLI::GetParam<string>("type");
  const size_t states = CLI::GetParam<int>("states");
  const double tolerance = CLI::GetParam<double>("tolerance");
  const bool batch = CLI::HasParam("batch");

  // Verify that either a model or a type was given.
  if (modelFile == "" && type == "")
    Log::Fatal << "No model file specified and no HMM type given!  At least "
        << "one is required." << endl;

  // If no model is specified, make sure we are training with valid parameters.
  if (modelFile == "")
  {
    // Validate number of states.
    if (states == 0)
      Log::Fatal << "Must specify number of states if model file is not "
          << "specified!" << endl;
  }

  if (modelFile != "" && CLI::HasParam("tolerance"))
    Log::Info << "Tolerance of existing model in '" << modelFile << "' will be "
        << "replaced with specified tolerance of " << tolerance << "." << endl;

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

  // If we have a model file, we can autodetect the type.
  if (CLI::HasParam("model_file"))
  {
    LoadHMMAndPerformAction<Train>(modelFile, &trainSeq);
  }
  else
  {
    // We need to read in the type and build the HMM by hand.
    const string type = CLI::GetParam<string>("type");

    if (type == "discrete")
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

      // Create HMM object.
      HMM<DiscreteDistribution> hmm(size_t(states),
          DiscreteDistribution(maxEmission), tolerance);

      // Initialize with random starting point.
      if (CLI::HasParam("random_initialization"))
      {
        hmm.Transition().randu();
        for (size_t c = 0; c < hmm.Transition().n_cols; ++c)
          hmm.Transition().col(c) /= arma::accu(hmm.Transition().col(c));

        for (size_t e = 0; e < hmm.Emission().size(); ++e)
        {
          hmm.Emission()[e].Probabilities().randu();
          hmm.Emission()[e].Probabilities() /=
              arma::accu(hmm.Emission()[e].Probabilities());
        }
      }

      // Now train it.  Pass the already-loaded training data.
      Train::Apply(hmm, &trainSeq);
    }
    else if (type == "gaussian")
    {
      // Find dimension of the data.
      const size_t dimensionality = trainSeq[0].n_rows;

      // Verify dimensionality of data.
      for (size_t i = 0; i < trainSeq.size(); ++i)
        if (trainSeq[i].n_rows != dimensionality)
          Log::Fatal << "Observation sequence " << i << " dimensionality ("
              << trainSeq[i].n_rows << " is incorrect (should be "
              << dimensionality << ")!" << endl;

      HMM<GaussianDistribution> hmm(size_t(states),
          GaussianDistribution(dimensionality), tolerance);

      // Initialize with random starting point.
      if (CLI::HasParam("random_initialization"))
      {
        hmm.Transition().randu();
        for (size_t c = 0; c < hmm.Transition().n_cols; ++c)
          hmm.Transition().col(c) /= arma::accu(hmm.Transition().col(c));

        for (size_t e = 0; e < hmm.Emission().size(); ++e)
        {
          hmm.Emission()[e].Mean().randu();
          // Generate random covariance.
          arma::mat r = arma::randu<arma::mat>(dimensionality, dimensionality);
          hmm.Emission()[e].Covariance(r * r.t());
        }
      }

      // Now train it.
      Train::Apply(hmm, &trainSeq);
    }
    else if (type == "gmm")
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
      HMM<GMM> hmm(size_t(states), GMM(size_t(gaussians), dimensionality),
          tolerance);

      // Initialize with random starting point.
      if (CLI::HasParam("random_initialization"))
      {
        hmm.Transition().randu();
        for (size_t c = 0; c < hmm.Transition().n_cols; ++c)
          hmm.Transition().col(c) /= arma::accu(hmm.Transition().col(c));

        for (size_t e = 0; e < hmm.Emission().size(); ++e)
        {
          // Random weights.
          hmm.Emission()[e].Weights().randu();
          hmm.Emission()[e].Weights() /=
              arma::accu(hmm.Emission()[e].Weights());

          // Random means and covariances.
          for (int g = 0; g < gaussians; ++g)
          {
            hmm.Emission()[e].Component(g).Mean().randu();

            // Generate random covariance.
            arma::mat r = arma::randu<arma::mat>(dimensionality,
                dimensionality);
            hmm.Emission()[e].Component(g).Covariance(r * r.t());
          }
        }
      }

      // Issue a warning if the user didn't give labels.
      if (!CLI::HasParam("labels_file"))
        Log::Warn << "Unlabeled training of GMM HMMs is almost certainly not "
            << "going to produce good results!" << endl;

      Train::Apply(hmm, &trainSeq);
    }
    else
    {
      Log::Fatal << "Unknown HMM type: " << type << "; must be 'discrete', "
          << "'gaussian', or 'gmm'." << endl;
    }
  }
}
