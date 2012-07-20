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

PROGRAM_INFO("Density estimation with DET", "This program provides an example "
    "use of the Density Estimation Tree for density estimation. For more "
    "details, please look at the paper titled 'Density Estimation Trees'.");

// Input data files.
PARAM_STRING_REQ("input/training_set", "The data set on which to perform "
    "density estimation.", "S");
PARAM_STRING("input/test_set", "An extra set of test points on which to "
    "estimate the density given the estimator.", "T", "");
PARAM_STRING("input/labels", "The labels for the given training data to "
    "generate the class membership of each leaf (as an extra statistic)", "L",
    "");

// Output data files.
PARAM_STRING("output/unpruned_tree_estimates", "The file in which to output the"
    " estimates on the training set from the large unpruned tree.", "u", "");
PARAM_STRING("output/training_set_estimates", "The file in which to output the "
    "estimates on the training set from the final optimally pruned tree.", "s",
    "");
PARAM_STRING("output/test_set_estimates", "The file in which to output the "
    "estimates on the test set from the final optimally pruned tree.", "t", "");
PARAM_STRING("output/leaf_class_table", "The file in which to output the leaf "
    "class membership table.", "l", "leaf_class_membership.txt");
PARAM_STRING("output/tree", "The file in which to print the final optimally "
    "pruned tree.", "p", "");
PARAM_STRING("output/vi", "The file to output the variable importance values "
    "for each feature.", "i", "");

// Parameters for the algorithm.
PARAM_INT("param/number_of_classes", "The number of classes present in the "
    "'labels' set provided", "C", 0);
PARAM_INT("param/folds", "The number of folds of cross-validation to perform "
    "for the estimation (enter 0 for LOOCV)", "F", 10);
PARAM_INT("DET/min_leaf_size", "The minimum size of a leaf in the unpruned "
    "fully grown DET.", "N", 5);
PARAM_INT("DET/max_leaf_size", "The maximum size of a leaf in the unpruned "
    "fully grown DET.", "M", 10);
PARAM_FLAG("DET/use_volume_reg", "This flag gives the used the option to use a "
    "form of regularization similar to the usual alpha-pruning in decision "
    "tree. But instead of regularizing on the number of leaves, you regularize "
    "on the sum of the inverse of the volume of the leaves (meaning you "
    "penalize low volume leaves.", "R");

// Some flags for output of some information about the tree.
PARAM_FLAG("flag/print_tree", "Print the tree out on the command line.", "P");
PARAM_FLAG("flag/print_vi", "Print the variable importance of each feature "
    "out on the command line.", "I");

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);

  string trainSetFile = CLI::GetParam<string>("input/training_set");
  arma::Mat<double> trainingData;

  data::Load(trainSetFile, trainingData, true);

  // Cross-validation here.
  size_t folds = CLI::GetParam<int>("param/folds");
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
      CLI::GetParam<string>("output/unpruned_tree_estimates");
  const bool regularization = CLI::HasParam("DET/use_volume_reg");
  const int maxLeafSize = CLI::GetParam<int>("DET/max_leaf_size");
  const int minLeafSize = CLI::GetParam<int>("DET/min_leaf_size");

  // Obtain the optimal tree.
  Timer::Start("det_training");
  DTree<double> *dtreeOpt = Trainer<double>(&trainingData, folds,
      regularization, maxLeafSize, minLeafSize, unprunedTreeEstimateFile);
  Timer::Stop("det_training");

  // Compute densities for the training points in the optimal tree.
  FILE *fp = NULL;

  if (CLI::GetParam<string>("output/training_set_estimates") != "")
  {
    fp = fopen(CLI::GetParam<string>("output/training_set_estimates").c_str(),
        "w");
  }

  // Computation timing is more accurate when printing is not performed.
  Timer::Start("det_estimation_time");
  for (size_t i = 0; i < trainingData.n_cols; i++)
  {
    arma::vec testPoint = trainingData.unsafe_col(i);
    double f = dtreeOpt->ComputeValue(testPoint);

    if (fp != NULL)
      fprintf(fp, "%lg\n", f);
  }
  Timer::Stop("det_estimation_time");

  if (fp != NULL)
    fclose(fp);

  // Compute the density at the provided test points and output the density in
  // the given file.
  if (CLI::GetParam<string>("input/test_set") != "")
  {
    const string testFile = CLI::GetParam<string>("input/test_set");
    arma::mat testData;
    data::Load(testFile, testData, true);

    fp = NULL;

    if (CLI::GetParam<string>("output/test_set_estimates") != "")
    {
      fp = fopen(CLI::GetParam<string>("output/test_set_estimates").c_str(),
          "w");
    }

    Timer::Start("det_test_set_estimation");
    for (size_t i = 0; i < testData.n_cols; i++)
    {
      arma::vec testPoint = testData.unsafe_col(i);
      double f = dtreeOpt->ComputeValue(testPoint);

      if (fp != NULL)
        fprintf(fp, "%lg\n", f);
    }
    Timer::Stop("det_test_set_estimation");

    if (fp != NULL)
      fclose(fp);
  }

  // Print the final tree.
  if (CLI::HasParam("flag/print_tree"))
  {
    fp = NULL;
    if (CLI::GetParam<string>("output/tree") != "")
    {
      fp = fopen(CLI::GetParam<string>("output/tree").c_str(), "w");

      if (fp != NULL)
      {
        dtreeOpt->WriteTree(0, fp);
        fclose(fp);
      }
    }
    else
    {
      dtreeOpt->WriteTree(0, stdout);
      printf("\n");
    }
  }

  // Print the leaf memberships for the optimal tree.
  if (CLI::GetParam<string>("input/labels") != "")
  {
    std::string labelsFile = CLI::GetParam<string>("input/labels");
    arma::Mat<size_t> labels;

    data::Load(labelsFile, labels, true);

    size_t num_classes = CLI::GetParam<int>("param/number_of_classes");
    if (num_classes == 0)
    {
      Log::Fatal << "Number of classes (param/number_of_classes) not specified!"
          << endl;
    }

    Log::Assert(trainingData.n_cols == labels.n_cols);
    Log::Assert(labels.n_rows == 1);

    PrintLeafMembership<double>(dtreeOpt, trainingData, labels, num_classes,
       CLI::GetParam<string>("output/leaf_class_table"));
  }

  // Print variable importance.
  if (CLI::HasParam("flag/print_vi"))
  {
    PrintVariableImportance<double>(dtreeOpt,
        CLI::GetParam<string>("output/vi"));
  }

  delete dtreeOpt;
}
