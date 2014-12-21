/**
 * @file dt_main.cpp
 * @ Parikshit Ram (pram@cc.gatech.edu)
 *
 * This file provides an example use of the DET
 *
 * This file is part of MLPACK 1.0.9.
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
#include "dt_utils.hpp"

using namespace mlpack;
using namespace mlpack::det;
using namespace std;

PROGRAM_INFO("Density Estimation With Density Estimation Trees",
    "This program performs a number of functions related to Density Estimation "
    "Trees.  The optimal Density Estimation Tree (DET) can be trained on a set "
    "of data (specified by --train_file) using cross-validation (with number of"
    " folds specified by --folds).  In addition, the density of a set of test "
    "points (specified by --test_file) can be estimated, and the importance of "
    "each dimension can be computed.  If class labels are given for the "
    "training points (with --labels_file), the class memberships of each leaf "
    "in the DET can be calculated."
    "\n\n"
    "The created DET can be saved to a file, along with the density estimates "
    "for the test set and the variable importances.");

// Input data files.
PARAM_STRING_REQ("train_file", "The data set on which to build a density "
    "estimation tree.", "t");
PARAM_STRING("test_file", "A set of test points to estimate the density of.",
    "T", "");
PARAM_STRING("labels_file", "The labels for the given training data to "
    "generate the class membership of each leaf (as an extra statistic)", "l",
    "");

// Output data files.
PARAM_STRING("unpruned_tree_estimates_file", "The file in which to output the "
    "density estimates on the training set from the large unpruned tree.", "u",
    "");
PARAM_STRING("training_set_estimates_file", "The file in which to output the "
    "density estimates on the training set from the final optimally pruned "
    "tree.", "e", "");
PARAM_STRING("test_set_estimates_file", "The file in which to output the "
    "estimates on the test set from the final optimally pruned tree.", "E", "");
PARAM_STRING("leaf_class_table_file", "The file in which to output the leaf "
    "class membership table.", "L", "leaf_class_membership.txt");
PARAM_STRING("tree_file", "The file in which to print the final optimally "
    "pruned tree.", "r", "");
PARAM_STRING("vi_file", "The file to output the variable importance values "
    "for each feature.", "i", "");

// Parameters for the algorithm.
PARAM_INT("folds", "The number of folds of cross-validation to perform for the "
    "estimation (0 is LOOCV)", "f", 10);
PARAM_INT("min_leaf_size", "The minimum size of a leaf in the unpruned, fully "
    "grown DET.", "N", 5);
PARAM_INT("max_leaf_size", "The maximum size of a leaf in the unpruned, fully "
    "grown DET.", "M", 10);
/*
PARAM_FLAG("volume_regularization", "This flag gives the used the option to use"
    "a form of regularization similar to the usual alpha-pruning in decision "
    "tree. But instead of regularizing on the number of leaves, you regularize "
    "on the sum of the inverse of the volume of the leaves (meaning you "
    "penalize low volume leaves.", "R");
*/

// Some flags for output of some information about the tree.
PARAM_FLAG("print_tree", "Print the tree out on the command line (or in the "
    "file specified with --tree_file).", "p");
PARAM_FLAG("print_vi", "Print the variable importance of each feature out on "
    "the command line (or in the file specified with --vi_file).", "I");

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);

  string trainSetFile = CLI::GetParam<string>("train_file");
  arma::Mat<double> trainingData;

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

  const string unprunedTreeEstimateFile =
      CLI::GetParam<string>("unpruned_tree_estimates_file");
  const bool regularization = false;
//  const bool regularization = CLI::HasParam("volume_regularization");
  const int maxLeafSize = CLI::GetParam<int>("max_leaf_size");
  const int minLeafSize = CLI::GetParam<int>("min_leaf_size");

  // Obtain the optimal tree.
  Timer::Start("det_training");
  DTree *dtreeOpt = Trainer(trainingData, folds, regularization, maxLeafSize,
      minLeafSize, unprunedTreeEstimateFile);
  Timer::Stop("det_training");

  // Compute densities for the training points in the optimal tree.
  FILE *fp = NULL;

  if (CLI::GetParam<string>("training_set_estimates_file") != "")
  {
    fp = fopen(CLI::GetParam<string>("training_set_estimates_file").c_str(),
        "w");

    // Compute density estimates for each point in the training set.
    Timer::Start("det_estimation_time");
    for (size_t i = 0; i < trainingData.n_cols; i++)
      fprintf(fp, "%lg\n", dtreeOpt->ComputeValue(trainingData.unsafe_col(i)));
    Timer::Stop("det_estimation_time");

    fclose(fp);
  }

  // Compute the density at the provided test points and output the density in
  // the given file.
  const string testFile = CLI::GetParam<string>("test_file");
  if (testFile != "")
  {
    arma::mat testData;
    data::Load(testFile, testData, true);

    fp = NULL;

    if (CLI::GetParam<string>("test_set_estimates_file") != "")
    {
      fp = fopen(CLI::GetParam<string>("test_set_estimates_file").c_str(), "w");

      Timer::Start("det_test_set_estimation");
      for (size_t i = 0; i < testData.n_cols; i++)
        fprintf(fp, "%lg\n", dtreeOpt->ComputeValue(testData.unsafe_col(i)));
      Timer::Stop("det_test_set_estimation");

      fclose(fp);
    }
  }

  // Print the final tree.
  if (CLI::HasParam("print_tree"))
  {
    fp = NULL;
    if (CLI::GetParam<string>("tree_file") != "")
    {
      fp = fopen(CLI::GetParam<string>("tree_file").c_str(), "w");

      if (fp != NULL)
      {
        dtreeOpt->WriteTree(fp);
        fclose(fp);
      }
    }
    else
    {
      dtreeOpt->WriteTree(stdout);
      printf("\n");
    }
  }

  // Print the leaf memberships for the optimal tree.
  if (CLI::GetParam<string>("labels_file") != "")
  {
    std::string labelsFile = CLI::GetParam<string>("labels_file");
    arma::Mat<size_t> labels;

    data::Load(labelsFile, labels, true);

    size_t numClasses = 0;
    for (size_t i = 0; i < labels.n_elem; ++i)
    {
      if (labels[i] > numClasses)
        numClasses = labels[i];
    }

    Log::Info << numClasses << " found in labels file '" << labelsFile << "'."
        << std::endl;

    Log::Assert(trainingData.n_cols == labels.n_cols);
    Log::Assert(labels.n_rows == 1);

    PrintLeafMembership(dtreeOpt, trainingData, labels, numClasses,
       CLI::GetParam<string>("leaf_class_table_file"));
  }

  // Print variable importance.
  if (CLI::HasParam("print_vi"))
  {
    PrintVariableImportance(dtreeOpt, CLI::GetParam<string>("vi_file"));
  }

  delete dtreeOpt;
}
