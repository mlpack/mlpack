/**
 * @file linear_regression_main.cpp
 * @author James Cline
 *
 * Main function for least-squares linear regression.
 */
#include <mlpack/core.hpp>
#include "linear_regression.hpp"

using namespace mlpack;

PARAM_STRING_REQ("train", "A file containing X", "linear_regression");
PARAM_STRING_REQ("test", "A file containing data points to predict on",
    "linear_regression");
PARAM_STRING("responses", "A file containing the y values for X; if not "
    "present, it is assumed the last column of train contains these values.",
    "linear_regression", "");

PARAM_MODULE("linear_regression",
    "Ordinary least squares linear regression: y = BX");

PROGRAM_INFO("Simple Linear Regression", "An implementation of simple linear "
    "regression using ordinary least squares.", "linear_regression");

int main(int argc, char* argv[])
{
  arma::vec B;
  arma::colvec responses;
  arma::mat predictors, file, points;

  // Handle parameters
  CLI::ParseCommandLine(argc, argv);

  const std::string train_name =
      CLI::GetParam<std::string>("linear_regression/train");
  const std::string test_name =
      CLI::GetParam<std::string>("linear_regression/test");
  const std::string response_name =
      CLI::GetParam<std::string>("linear_regression/responses");

  data::Load(train_name.c_str(), file, true);
  size_t n_cols = file.n_cols,
         n_rows = file.n_rows;

  if (response_name == "")
  {
    predictors = file.submat(0,0, n_rows-2, n_cols-1);
    // The initial predictors for y, Nx1
    responses = arma::trans(file.row(n_rows-1));
    --n_rows;
  }
  else
  {
    predictors = file;
    // The initial predictors for y, Nx1
    data::Load(response_name.c_str(), responses, true);

    if (responses.n_rows > 1)
      Log::Fatal << "The responses must have one column.\n";

    if (responses.n_cols != n_cols)
      Log::Fatal << "The responses must have the same number of rows as the "
          "training file.\n";
  }

  data::Load(test_name.c_str(), points, true);

  if (points.n_rows != n_rows)
    Log::Fatal << "The test data must have the same number of columns as the "
        "training file.\n";

  arma::rowvec predictions;

  linear_regression::LinearRegression lr(predictors, responses);
  lr.predict(predictions, points);

  //data.row(n_rows) = predictions;
  //data::Save("out.csv", data);
  //std::cout << "predictions: " << arma::trans(predictions) << '\n';

  return 0;
}
