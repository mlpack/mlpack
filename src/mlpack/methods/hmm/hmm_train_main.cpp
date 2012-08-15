/**
 * @file hmm_train_main.cpp
 * @author Ryan Curtin
 *
 * Executable which trains an HMM and saves the trained HMM to file.
 * This file is part of MLPACK 1.0.2.
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
#include <mlpack/core.hpp>

#include "hmm.hpp"
#include "hmm_util.hpp"

#include <mlpack/methods/gmm/gmm.hpp>

PROGRAM_INFO("Hidden Markov Model (HMM) Training", "This program allows a "
    "Hidden Markov Model to be trained on labeled or unlabeled data.  It "
    "support three types of HMMs: discrete HMMs, Gaussian HMMs, or GMM HMMs."
    "\n"
    "Either one input sequence can be specified (with --input_file), or, a "
    "file containing files in which input sequences can be found (when "
    "--input_file and --batch are used together).  In addition, labels can be "
    "provided in the file specified by --label_file, and if --batch is used, "
    "the file given to --label_file should contain a list of files of labels "
    "corresponding to the sequences in the file given to --input_file.\n"
    "\n"
    "Optionally, a pre-created HMM model can be used as a guess for the "
    "transition matrix and emission probabilities; this is specifiable with "
    "--model_file.");

PARAM_STRING_REQ("input_file", "File containing input observations.", "i");
PARAM_STRING_REQ("type", "Type of HMM: discrete | gaussian | gmm.", "t");

PARAM_FLAG("batch", "If true, input_file (and if passed, labels_file) are "
    "expected to contain a list of files to use as input observation sequences "
    " (and label sequences).", "b");
PARAM_INT("states", "Number of hidden states in HMM (necessary, unless "
    "model_file is specified.", "n", 0);
PARAM_INT("gaussians", "Number of gaussians in each GMM (necessary when type is"
    " 'gmm'.", "g", 0);
PARAM_STRING("model_file", "Pre-existing HMM model (optional).", "m", "");
PARAM_STRING("labels_file", "Optional file of hidden states, used for "
    "labeled training.", "l", "");
PARAM_STRING("output_file", "File to save trained HMM to (XML).", "o",
    "output_hmm.xml");
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;
using namespace mlpack::utilities;
using namespace mlpack::gmm;
using namespace mlpack::math;
using namespace arma;
using namespace std;

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
  const string labelsFile = CLI::GetParam<string>("labels_file");
  const string modelFile = CLI::GetParam<string>("model_file");
  const string outputFile = CLI::GetParam<string>("output_file");
  const string type = CLI::GetParam<string>("type");
  const int states = CLI::GetParam<int>("states");
  const bool batch = CLI::HasParam("batch");

  // Validate number of states.
  if (states == 0 && modelFile == "")
  {
    Log::Fatal << "Must specify number of states if model file is not "
        << "specified!" << endl;
  }

  if (states < 0 && modelFile == "")
  {
    Log::Fatal << "Invalid number of states (" << states << "); must be greater"
        << " than or equal to 1." << endl;
  }

  // Load the dataset(s) and labels.
  vector<mat> trainSeq;
  vector<arma::Col<size_t> > labelSeq; // May be empty.
  if (batch)
  {
    // The input file contains a list of files to read.
    Log::Info << "Reading list of training sequences from '" << inputFile
        << "'." << endl;

    fstream f(inputFile.c_str(), ios_base::in);

    if (!f.is_open())
      Log::Fatal << "Could not open '" << inputFile << "' for reading." << endl;

    // Now read each line in.
    char lineBuf[1024]; // Max 1024 characters... hopefully that is long enough.
    f.getline(lineBuf, 1024, '\n');
    while (!f.eof())
    {
      Log::Info << "Adding training sequence from '" << lineBuf << "'." << endl;

      // Now read the matrix.
      trainSeq.push_back(mat());
      if (labelsFile == "") // Nonfatal in this case.
      {
        if (!data::Load(lineBuf, trainSeq.back(), false))
        {
          Log::Warn << "Loading training sequence from '" << lineBuf << "' "
              << "failed.  Sequence ignored." << endl;
          trainSeq.pop_back(); // Remove last element which we did not use.
        }
      }
      else
      {
        data::Load(lineBuf, trainSeq.back(), true);
      }

      // See if we need to transpose the data.
      if (type == "discrete")
      {
        if (trainSeq.back().n_cols == 1)
          trainSeq.back() = trans(trainSeq.back());
      }

      f.getline(lineBuf, 1024, '\n');
    }

    f.close();

    // Now load labels, if we need to.
    if (labelsFile != "")
    {
      f.open(labelsFile.c_str(), ios_base::in);

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

        // Ensure that matrix only has one column.
        if (label.n_rows == 1)
          label = trans(label);

        if (label.n_cols > 1)
          Log::Fatal << "Invalid labels; must be one-dimensional." << endl;

        labelSeq.push_back(label.col(0));

        f.getline(lineBuf, 1024, '\n');
      }
    }
  }
  else
  {
    // Only one input file.
    trainSeq.resize(1);
    data::Load(inputFile.c_str(), trainSeq[0], true);

    // Do we need to load labels?
    if (labelsFile != "")
    {
      Mat<size_t> label;
      data::Load(labelsFile, label, true);

      // Ensure that matrix only has one column.
      if (label.n_rows == 1)
        label = trans(label);

      if (label.n_cols > 1)
        Log::Fatal << "Invalid labels; must be one-dimensional." << endl;

      // Verify the same number of observations as the data.
      if (label.n_elem != trainSeq[labelSeq.size()].n_cols)
        Log::Fatal << "Label sequence " << labelSeq.size() << " does not have "
            << "the same number of points as observation sequence "
            << labelSeq.size() << "!" << endl;

      labelSeq.push_back(label.col(0));
    }
  }

  // Now, train the HMM, since we have loaded the input data.
  if (type == "discrete")
  {
    // Verify observations are valid.
    for (size_t i = 0; i < trainSeq.size(); ++i)
      if (trainSeq[i].n_rows > 1)
        Log::Fatal << "Error in training sequence " << i << ": only "
            << "one-dimensional discrete observations allowed for discrete "
            << "HMMs!" << endl;

    // Do we have a model to preload?
    HMM<DiscreteDistribution> hmm(1, DiscreteDistribution(1));

    if (modelFile != "")
    {
      SaveRestoreUtility loader;
      loader.ReadFile(modelFile);
      LoadHMM(hmm, loader);
    }
    else // New model.
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
      hmm = HMM<DiscreteDistribution>(size_t(states),
          DiscreteDistribution(maxEmission));
    }

    // Do we have labels?
    if (labelsFile == "")
      hmm.Train(trainSeq); // Unsupervised training.
    else
      hmm.Train(trainSeq, labelSeq); // Supervised training.

    // Finally, save the model.  This should later be integrated into the HMM
    // class itself.
    SaveRestoreUtility sr;
    SaveHMM(hmm, sr);
    sr.WriteFile(outputFile);
  }
  else if (type == "gaussian")
  {
    // Create HMM object.
    HMM<GaussianDistribution> hmm(1, GaussianDistribution(1));

    // Do we have a model to load?
    size_t dimensionality = 0;
    if (modelFile != "")
    {
      SaveRestoreUtility loader;
      loader.ReadFile(modelFile);
      LoadHMM(hmm, loader);

      dimensionality = hmm.Emission()[0].Mean().n_elem;
    }
    else
    {
      // Find dimension of the data.
      dimensionality = trainSeq[0].n_rows;

      hmm = HMM<GaussianDistribution>(size_t(states),
          GaussianDistribution(dimensionality));
    }

    // Verify dimensionality of data.
    for (size_t i = 0; i < trainSeq.size(); ++i)
      if (trainSeq[i].n_rows != dimensionality)
        Log::Fatal << "Observation sequence " << i << " dimensionality ("
            << trainSeq[i].n_rows << " is incorrect (should be "
            << dimensionality << ")!" << endl;

    // Now run the training.
    if (labelsFile == "")
      hmm.Train(trainSeq); // Unsupervised training.
    else
      hmm.Train(trainSeq, labelSeq); // Supervised training.

    // Finally, save the model.  This should later be integrated into th HMM
    // class itself.
    SaveRestoreUtility sr;
    SaveHMM(hmm, sr);
    sr.WriteFile(outputFile);
  }
  else if (type == "gmm")
  {
    // Create HMM object.
    HMM<GMM<> > hmm(1, GMM<>(1, 1));

    // Do we have a model to load?
    size_t dimensionality = 0;
    if (modelFile != "")
    {
      SaveRestoreUtility loader;
      loader.ReadFile(modelFile);
      LoadHMM(hmm, loader);

      dimensionality = hmm.Emission()[0].Dimensionality();
    }
    else
    {
      // Find dimension of the data.
      dimensionality = trainSeq[0].n_rows;

      const int gaussians = CLI::GetParam<int>("gaussians");

      if (gaussians == 0)
        Log::Fatal << "Number of gaussians for each GMM must be specified (-g) "
            << "when type = 'gmm'!" << endl;

      if (gaussians < 0)
        Log::Fatal << "Invalid number of gaussians (" << gaussians << "); must "
            << "be greater than or equal to 1." << endl;

      hmm = HMM<GMM<> >(size_t(states), GMM<>(size_t(gaussians),
          dimensionality));
    }

    // Verify dimensionality of data.
    for (size_t i = 0; i < trainSeq.size(); ++i)
      if (trainSeq[i].n_rows != dimensionality)
        Log::Fatal << "Observation sequence " << i << " dimensionality ("
            << trainSeq[i].n_rows << " is incorrect (should be "
            << dimensionality << ")!" << endl;

    // Now run the training.
    if (labelsFile == "")
    {
      Log::Warn << "Unlabeled training of GMM HMMs is almost certainly not "
          << "going to produce good results!" << endl;
      hmm.Train(trainSeq);
    }
    else
    {
      hmm.Train(trainSeq, labelSeq);
    }

    // Save model.
    SaveRestoreUtility sr;
    SaveHMM(hmm, sr);
    sr.WriteFile(outputFile);
  }
  else
  {
    Log::Fatal << "Unknown HMM type: " << type << "; must be 'discrete', "
        << "'gaussian', or 'gmm'." << endl;
  }
}
