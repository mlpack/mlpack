/**
 * @file feature_selection_main.cpp
 * @author Jeffin Sam
 *
 * A CLI executable to select best features from a dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/feature_selection.hpp>
#include <mlpack/core/data/chi2_feature_selection.hpp>
#include <mlpack/core/data/correlation_feature_selection.hpp>

PROGRAM_INFO("Feature Selection",
    // Short description.
    "A utility to reduce dimesnonality of dataset by selecting best features"
    "from dataset using different methods.",
    // Long description.
    "This utility takes a dataset and reduces the dimesnonality by selecting "
    "the best features from it using either variance based, or Chi2 base or "
    "correlation based feature selection method. You can use " +
    PRINT_PARAM_STRING("method") + "parameter to specify the method to be used."
    "\n\n"
    "The output matrices with reduced dimesnonality may be saved with the " +
    PRINT_PARAM_STRING("output") + " parameters."
    "\n\n"
    "You can use variance based or correlation based feature selection method"
    " on continous data, and for dicrete data, chi2 method could be used."
    "\n\n"
    "So, a simple example where we want to select best feature from the" 
    " dataset " + PRINT_DATASET("X") + " into " + PRINT_DATASET("X_reduced")
    + " and use variance based feature selection method could be"
    "\n\n" +
    PRINT_CALL("feature_selection", "input", "X", "output", "X_reduced",
        "method", "variance", "threshold", 0.22) +
    "\n\n"
    "Another example where we want to select 4 best feature from the" +
    " dataset "+ PRINT_DATASET("X") + " into " + PRINT_DATASET("X_reduced")
    + " and use chi2 based feature selection method could be"
    "\n\n" +
    PRINT_CALL("feature_selection", "input", "X", "output_size", 4, "output",
        "X_reduced", "method", "chi2"),
    SEE_ALSO("@preprocess_binarize", "#preprocess_binarize"),
    SEE_ALSO("@preprocess_describe", "#preprocess_describe"),
    SEE_ALSO("@preprocess_imputer", "#preprocess_imputer"));

// Define parameters for data.
PARAM_MATRIX_IN_REQ("input", "Matrix containing data.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save selected features data to.", "o");

PARAM_STRING_IN_REQ("method", "Method used to selecte best feature",
    "m");
PARAM_DOUBLE_IN("threshold", "Threshold value for variance,"
    "based feature selection", "r", 0.05);

PARAM_INT_IN("output_size", "Size of the output matrix you want", "k", 1);
PARAM_INT_IN_REQ("labels_row", "Row number of the ouput labels", "l");

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

static void mlpackMain()
{

  // Load the data.
  arma::mat& data = CLI::GetParam<arma::mat>("input");
  const std::string& method = CLI::GetParam<std::string>("method");
  const int& labelsRow = CLI::GetParam<int>("labels_row");
  arma::rowvec labels = data.row(labelsRow);
  data.shed_row(labelsRow); 
  arma::mat output;
  if (method == "correlation")
  {
    const int& outputSize = CLI::GetParam<int>("output_size");
    data::fs::CorrelationSelection(data, labels, output, outputSize);
  }
  else if (method == "variance")
  {
    const double& threshold = CLI::GetParam<double>("threshold");
    data::fs::VarianceSelection(data, threshold, output);
  }
  else if (method == "chi2")
  {
    const int& outputSize = CLI::GetParam<int>("output_size");
    data::fs::Chi2Selection(data, labels, output, outputSize);    
  }
  else
  {
    Log::Fatal << "Method not recognized." << endl;
  }
  output.insert_rows(labelsRow, labels);
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(output);
}
