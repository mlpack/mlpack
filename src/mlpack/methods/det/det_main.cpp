/**
 * @file dt_main.cpp
 * @ Parikshit Ram (pram@cc.gatech.edu)
 *
 * This file provides an example use of the DET
 */

#include <mlpack/core.hpp>
#include "dt_utils.hpp"

using namespace mlpack;
using namespace mlpack::det;
using namespace std;

PROGRAM_INFO("Density Estimation With Density Estimation Trees",
    "This program performs a number of functions related to Density Estimation "
    "Trees.  The optimal Density Estimation Tree (DET) can be trained on a set "
    "of data (specified by --training_file or -t) using cross-validation (with "
    "number of folds specified by --folds).  This trained density estimation "
    "tree may then be saved to a model file with the --output_model_file (-M) "
    "option."
    "\n\n"
    "The variable importances of each dimension may be saved with the "
    "--vi_file (-i) option, and the density estimates on each training point "
    "may be saved to the file specified with the --training_set_estimates_file "
    "(-e) option."
    "\n\n"
    "This program also can provide density estimates for a set of test points, "
    "specified in the --test_file (-T) file.  The density estimation tree used "
    "for this task will be the tree that was trained on the given training "
    "points, or a tree stored in the file given with the --input_model_file "
    "(-m) parameter.  The density estimates for the test points may be saved "
    "into the file specified with the --test_set_estimates_file (-E) option.");

// Input data files.
PARAM_STRING("training_file", "The data set on which to build a density "
    "estimation tree.", "t", "");

// Input or output model.
PARAM_STRING("input_model_file", "File containing already trained density "
    "estimation tree.", "m", "");
PARAM_STRING("output_model_file", "File to save trained density estimation tree"
    " to.", "M", "");

// Output data files.
PARAM_STRING("test_file", "A set of test points to estimate the density of.",
    "T", "");
PARAM_STRING("training_set_estimates_file", "The file in which to output the "
    "density estimates on the training set from the final optimally pruned "
    "tree.", "e", "");
PARAM_STRING("test_set_estimates_file", "The file in which to output the "
    "estimates on the test set from the final optimally pruned tree.", "E", "");
PARAM_STRING("vi_file", "The file to output the variable importance values "
    "for each feature.", "i", "");

// Parameters for the training algorithm.
PARAM_INT("folds", "The number of folds of cross-validation to perform for the "
    "estimation (0 is LOOCV)", "f", 10);
PARAM_INT("min_leaf_size", "The minimum size of a leaf in the unpruned, fully "
    "grown DET.", "l", 5);
PARAM_INT("max_leaf_size", "The maximum size of a leaf in the unpruned, fully "
    "grown DET.", "L", 10);
/*
PARAM_FLAG("volume_regularization", "This flag gives the used the option to use"
    "a form of regularization similar to the usual alpha-pruning in decision "
    "tree. But instead of regularizing on the number of leaves, you regularize "
    "on the sum of the inverse of the volume of the leaves (meaning you "
    "penalize low volume leaves.", "R");
*/

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);

  // Validate input parameters.
  if (CLI::HasParam("training_file") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Only one of --training_file (-t) or --input_model_file (-m) "
        << "may be specified!" << endl;

  if (!CLI::HasParam("training_file") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "Neither --training_file (-t) nor --input_model_file (-m) "
        << "are specified!" << endl;

  if (!CLI::HasParam("training_file"))
  {
    if (CLI::HasParam("training_set_estimates_file"))
      Log::Warn << "--training_set_estimates_file (-e) ignored because "
          << "--training_file (-t) is not specified." << endl;
    if (CLI::HasParam("folds"))
      Log::Warn << "--folds (-f) ignored because --training_file (-t) is not "
          << "specified." << endl;
    if (CLI::HasParam("min_leaf_size"))
      Log::Warn << "--min_leaf_size (-l) ignored because --training_file (-t) "
          << "is not specified." << endl;
    if (CLI::HasParam("max_leaf_size"))
      Log::Warn << "--max_leaf_size (-L) ignored because --training_file (-t) "
          << "is not specified." << endl;
  }

  if (!CLI::HasParam("test_file") && CLI::HasParam("test_set_estimates_file"))
    Log::Warn << "--test_set_estimates_file (-E) ignored because --test_file "
        << "(-T) is not specified." << endl;

  // Are we training a DET or loading from file?
  DTree* tree;
  if (CLI::HasParam("training_file"))
  {
    const string trainSetFile = CLI::GetParam<string>("training_file");
    arma::mat trainingData;
    data::Load(trainSetFile, trainingData, true);

    // Cross-validation here.
    size_t folds = CLI::GetParam<int>("folds");
    if (folds == 0)
    {
      folds = trainingData.n_cols;
      Log::Info << "Performing leave-one-out cross validation." << endl;
    }
    else
    {
      Log::Info << "Performing " << folds << "-fold cross validation." << endl;
    }

    const bool regularization = false;
//    const bool regularization = CLI::HasParam("volume_regularization");
    const int maxLeafSize = CLI::GetParam<int>("max_leaf_size");
    const int minLeafSize = CLI::GetParam<int>("min_leaf_size");

    // Obtain the optimal tree.
    Timer::Start("det_training");
    tree = Trainer(trainingData, folds, regularization, maxLeafSize,
        minLeafSize, "");
    Timer::Stop("det_training");

    // Compute training set estimates, if desired.
    if (CLI::HasParam("training_set_estimates_file"))
    {
      // Compute density estimates for each point in the training set.
      arma::rowvec trainingDensities(trainingData.n_cols);
      Timer::Start("det_estimation_time");
      for (size_t i = 0; i < trainingData.n_cols; i++)
        trainingDensities[i] = tree->ComputeValue(trainingData.unsafe_col(i));
      Timer::Stop("det_estimation_time");

      data::Save(CLI::GetParam<string>("training_set_estimates_file"),
          trainingDensities);
    }
  }
  else
  {
    data::Load(CLI::GetParam<string>("input_model_file"), "det_model", tree,
        true);
  }

  // Compute the density at the provided test points and output the density in
  // the given file.
  const string testFile = CLI::GetParam<string>("test_file");
  if (testFile != "")
  {
    arma::mat testData;
    data::Load(testFile, testData, true);

    // Compute test set densities.
    Timer::Start("det_test_set_estimation");
    arma::rowvec testDensities(testData.n_cols);
    for (size_t i = 0; i < testData.n_cols; i++)
      testDensities[i] = tree->ComputeValue(testData.unsafe_col(i));
    Timer::Stop("det_test_set_estimation");

    if (CLI::GetParam<string>("test_set_estimates_file") != "")
      data::Save(CLI::GetParam<string>("test_set_estimates_file"),
          testDensities);
  }

  // Print variable importance.
  if (CLI::HasParam("vi_file"))
    PrintVariableImportance(tree, CLI::GetParam<string>("vi_file"));

  // Save the model, if desired.
  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<string>("output_model_file"), "det_model", tree,
        false);

  delete tree;
}
