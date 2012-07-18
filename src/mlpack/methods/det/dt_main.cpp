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
    "use of the Density Estimation "
	     "Tree for density estimation. For more details, "
	     "please look at the paper titled "
	     "'Density Estimation Trees'.");

// input data files
PARAM_STRING_REQ("input/training_set", "The data set on which to "
		 "perform density estimation.", "S");
PARAM_STRING("input/test_set", "An extra set of test points on "
	     "which to estimate the density given the estimator.",
	     "T", "");
PARAM_STRING("input/labels", "The labels for the given training data to "
	     "generate the class membership of each leaf (as an "
	     "extra statistic)", "L", "");

// output data files
PARAM_STRING("output/unpruned_tree_estimates", "The file "
	     "in which to output the estimates on the "
	     "training set from the large unpruned tree.", "u", "");
PARAM_STRING("output/training_set_estimates", "The file "
	     "in which to output the estimates on the "
	     "training set from the final optimally pruned"
	     " tree.", "s", "");
PARAM_STRING("output/test_set_estimates", "The file "
	     "in which to output the estimates on the "
	     "test set from the final optimally pruned"
	     " tree.", "t", "");
PARAM_STRING("output/leaf_class_table", "The file "
	     "in which to output the leaf class membership "
	     "table.", "l", "leaf_class_membership.txt");
PARAM_STRING("output/tree", "The file in which to print "
	     "the final optimally pruned tree.", "p", "");
PARAM_STRING("output/vi", "The file to output the "
	     "variable importance values for each feature.",
	     "i", "");

// parameters for the algorithm
PARAM_INT("param/number_of_classes", "The number of classes present "
	  "in the 'labels' set provided", "C", 0);
PARAM_INT("param/folds", "The number of folds of cross-validation"
	  " to performed for the estimation (enter 0 for LOOCV)",
	  "F", 10);
PARAM_INT("DET/min_leaf_size", "The minimum size of a leaf"
	  " in the unpruned fully grown DET.", "N", 5);
PARAM_INT("DET/max_leaf_size", "The maximum size of a leaf"
	  " in the unpruned fully grown DET.", "M", 10);
PARAM_FLAG("DET/use_volume_reg", "This flag gives the used the "
	   "option to use a form of regularization similar to "
	   "the usual alpha-pruning in decision tree. But "
	   "instead of regularizing on the number of leaves, "
	   "you regularize on the sum of the inverse of the volume "
	   "of the leaves (meaning you penalize  "
	   "low volume leaves.", "R");

// some flags for output of some information about the tree
PARAM_FLAG("flag/print_tree", "If you just wish to print the tree "
	   "out on the command line.", "P");
PARAM_FLAG("flag/print_vi", "If you just wish to print the "
	   "variable importance of each feature "
	   "out on the command line.", "I");

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);

  string train_set_file = CLI::GetParam<string>("S");
  arma::Mat<double> training_data;

  Log::Info << "Loading training set..." << endl;
  if (!data::Load(train_set_file, training_data))
    Log::Fatal << "Training set file "<< train_set_file
	       << " can't be loaded." << endl;

  Log::Info << "Training set (" << training_data.n_rows
	    << ", " << training_data.n_cols
	    << ")" << endl;

  // cross-validation here
  size_t folds = CLI::GetParam<int>("F");
  if (folds == 0) {
    folds = training_data.n_cols;
    Log::Info << "Starting Leave-One-Out Cross validation" << endl;
  } else
    Log::Info << "Starting " << folds
	      << "-fold Cross validation" << endl;



  // obtaining the optimal tree
  string unpruned_tree_estimate_file
    = CLI::GetParam<string>("u");

  Timer::Start("DET/Training");
  DTree<double> *dtree_opt = Trainer<double>
    (&training_data, folds, CLI::HasParam("R"), CLI::GetParam<int>("M"),
     CLI::GetParam<int>("N"), unpruned_tree_estimate_file);
  Timer::Stop("DET/Training");

  // computing densities for the train points in the
  // optimal tree
  FILE *fp = NULL;

  if (CLI::GetParam<string>("s") != "") {
    string optimal_estimates_file = CLI::GetParam<string>("s");
    fp = fopen(optimal_estimates_file.c_str(), "w");
  }

  // Computation timing is more accurate when you do not
  // perform the printing.
  Timer::Start("DET/EstimationTime");
  for (size_t i = 0; i < training_data.n_cols; i++) {
    arma::Col<double> test_p = training_data.unsafe_col(i);
    long double f = dtree_opt->ComputeValue(test_p);
    if (fp != NULL)
      fprintf(fp, "%Lg\n", f);
  } // end for
  Timer::Stop("DET/EstimationTime");

  if (fp != NULL)
    fclose(fp);


  // computing the density at the provided test points
  // and outputting the density in the given file.
  if (CLI::GetParam<string>("T") != "") {
    string test_file = CLI::GetParam<string>("T");
    arma::Mat<double> test_data;
    Log::Info << "Loading test set..." << endl;
    if (!data::Load(test_file, test_data))
      Log::Fatal << "Test set file "<< test_file
		 << " can't be loaded." << endl;

    Log::Info << "Test set (" << test_data.n_rows
	      << ", " << test_data.n_cols
	      << ")" << endl;

    fp = NULL;

    if (CLI::GetParam<string>("t") != "") {
      string test_density_file
	= CLI::GetParam<string>("t");
      fp = fopen(test_density_file.c_str(), "w");
    }

    Timer::Start("DET/TestSetEstimation");
    for (size_t i = 0; i < test_data.n_cols; i++) {
      arma::Col<double> test_p = test_data.unsafe_col(i);
      long double f = dtree_opt->ComputeValue(test_p);
      if (fp != NULL)
	fprintf(fp, "%Lg\n", f);
    } // end for
    Timer::Stop("DET/TestSetEstimation");

    if (fp != NULL)
      fclose(fp);
  } // Test set estimation

  // printing the final tree
  if (CLI::HasParam("P")) {

    fp = NULL;
    if (CLI::GetParam<string>("p") != "") {
      string print_tree_file = CLI::GetParam<string>("p");
      fp = fopen(print_tree_file.c_str(), "w");

      if (fp != NULL) {
	dtree_opt->WriteTree(0, fp);
	fclose(fp);
      }
    } else {
      dtree_opt->WriteTree(0, stdout);
      printf("\n");
    }
  } // Printing the tree

  // print the leaf memberships for the optimal tree
  if (CLI::GetParam<string>("L") != "") {
    std::string labels_file = CLI::GetParam<string>("L");
    arma::Mat<int> labels;

    Log::Info << "Loading label file..." << endl;
    if (!data::Load(labels_file, labels))
      Log::Fatal << "Label file "<< labels_file
		 << " can't be loaded." << endl;

    Log::Info << "Labels (" << labels.n_rows
	      << ", " << labels.n_cols
	      << ")" << endl;

    size_t num_classes = CLI::GetParam<int>("C");
    if (num_classes == 0)
      Log::Fatal << "Please provide the number of classes"
		 << " present in the label file" << endl;

    assert(training_data.n_cols == labels.n_cols);
    assert(labels.n_rows == 1);

    PrintLeafMembership<double>
      (dtree_opt, training_data, labels, num_classes,
       (string) CLI::GetParam<string>("l"));
  } // leaf class membership


  if(CLI::HasParam("I")) {
    PrintVariableImportance<double>
      (dtree_opt, training_data.n_rows,
       (string) CLI::GetParam<string>("i"));
  } // print variable importance


  delete dtree_opt;
  return 0;
} // end main
