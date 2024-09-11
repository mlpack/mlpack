/**
 * @file methods/preprocess/preprocess_describe_main.cpp
 * @author Keon Kim
 *
 * Descriptive Statistics Class and binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME preprocess_describe

#include <mlpack/core/util/mlpack_main.hpp>

#include <iomanip>

using namespace mlpack;
using namespace mlpack::data;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("Descriptive Statistics");

// Short description.
BINDING_SHORT_DESC(
    "A utility for printing descriptive statistics about a dataset.  This "
    "prints a number of details about a dataset in a tabular format.");

// Long description.
BINDING_LONG_DESC(
    "This utility takes a dataset and prints out the descriptive statistics "
    "of the data. Descriptive statistics is the discipline of quantitatively "
    "describing the main features of a collection of information, or the "
    "quantitative description itself. The program does not modify the original "
    "file, but instead prints out the statistics to the console. The printed "
    "result will look like a table."
    "\n\n"
    "Optionally, width and precision of the output can be adjusted by a user "
    "using the " + PRINT_PARAM_STRING("width") + " and " +
    PRINT_PARAM_STRING("precision") + " parameters. A user can also select a "
    "specific dimension to analyze if there are too many dimensions. The " +
    PRINT_PARAM_STRING("population") + " parameter can be specified when the "
    "dataset should be considered as a population.  Otherwise, the dataset "
    "will be considered as a sample.");

// Example.
BINDING_EXAMPLE(
    "So, a simple example where we want to print out statistical facts about "
    "the dataset " + PRINT_DATASET("X") + " using the default settings, we "
    "could run "
    "\n\n" +
    PRINT_CALL("preprocess_describe", "input", "X", "verbose", true) +
    "\n\n"
    "If we want to customize the width to 10 and precision to 5 and consider "
    "the dataset as a population, we could run"
    "\n\n" +
    PRINT_CALL("preprocess_describe", "input", "X", "width", 10, "precision", 5,
        "verbose", true));

// See also...
BINDING_SEE_ALSO("@preprocess_binarize", "#preprocess_binarize");
#if BINDING_TYPE == BINDING_TYPE_CLI
BINDING_SEE_ALSO("@preprocess_imputer", "#preprocess_imputer");
#endif
BINDING_SEE_ALSO("@preprocess_split", "#preprocess_split");

// Define parameters for data.
PARAM_MATRIX_IN_REQ("input", "Matrix containing data,", "i");
PARAM_INT_IN("dimension", "Dimension of the data. Use this to specify a "
    "dimension", "d", 0);
PARAM_INT_IN("precision", "Precision of the output statistics.", "p", 4);
PARAM_INT_IN("width", "Width of the output table.", "w", 8);
PARAM_FLAG("population", "If specified, the program will calculate statistics "
    "assuming the dataset is the population. By default, the program will "
    "assume the dataset as a sample.", "P");
PARAM_FLAG("row_major", "If specified, the program will calculate statistics "
    "across rows, not across columns.  (Remember that in mlpack, a column "
    "represents a point, so this option is generally not necessary.)", "r");

/**
 * Calculates the sum of deviations to the Nth Power.
 *
 * @param input Vector that captures a dimension of a dataset.
 * @param rowMean Mean of the given vector.
 * @param n Degree of power.
 * @return sum of nth power deviations.
 */
double SumNthPowerDeviations(const arma::rowvec& input,
                             const double& fMean,
                             size_t n)
{
  return sum(pow(input - fMean, static_cast<double>(n)));
}

/**
 * Calculates Skewness of the given vector.
 *
 * @param input Vector that captures a dimension of a dataset
 * @param rowStd Standard Deviation of the given vector.
 * @param rowMean Mean of the given vector.
 * @return Skewness of the given vector.
 */
double Skewness(const arma::rowvec& input,
                const double& fStd,
                const double& fMean,
                const bool population)
{
  double skewness = 0;
  const double S3 = pow(fStd, 3);
  const double M3 = SumNthPowerDeviations(input, fMean, 3);
  const double n = input.n_elem;
  if (population)
  {
    // Calculate population skewness
    skewness = M3 / (n * S3);
  }
  else
  {
    // Calculate sample skewness.
    skewness = n * M3 / ((n - 1) * (n - 2) * S3);
  }
  return skewness;
}

/**
 * Calculates excess kurtosis of the given vector.
 *
 * @param input Vector that captures a dimension of a dataset
 * @param rowStd Standard Deviation of the given vector.
 * @param rowMean Mean of the given vector.
 * @return Kurtosis of the given vector.
 */
double Kurtosis(const arma::rowvec& input,
                const double& fStd,
                const double& fMean,
                const bool population)
{
  double kurtosis = 0;
  const double M4 = SumNthPowerDeviations(input, fMean, 4);
  const double n = input.n_elem;
  if (population)
  {
    // Calculate population excess kurtosis.
    const double M2 = SumNthPowerDeviations(input, fMean, 2);
    kurtosis = n * (M4 / pow(M2, 2)) - 3;
  }
  else
  {
    // Calculate sample excess kurtosis.
    const double S4 = pow(fStd, 4);
    const double norm3 = (3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3));
    const double normC = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3));
    const double normM = M4 / S4;
    kurtosis = normC * normM - norm3;
  }
  return kurtosis;
}

/**
 * Calculates standard error of standard deviation.
 *
 * @param input Vector that captures a dimension of a dataset
 * @param rowStd Standard Deviation of the given vector.
 * @return Standard error of the stanrdard devation of the given vector.
 */
double StandardError(const size_t size, const double& fStd)
{
  return fStd / sqrt(size);
}

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  const size_t dimension = static_cast<size_t>(params.Get<int>("dimension"));
  const size_t precision = static_cast<size_t>(params.Get<int>("precision"));
  const size_t width = static_cast<size_t>(params.Get<int>("width"));
  const bool population = params.Has("population");
  const bool rowMajor = params.Has("row_major");

  // Load the data.
  arma::mat& data = params.Get<arma::mat>("input");

  timers.Start("statistics");
  // Print the headers.
  Log::Info << setw(width) << "dim" << setw(width) << "var" << setw(width)
      << "mean" << setw(width) << "std" << setw(width)
      << "median" << setw(width) << "min" << setw(width)
      << "max" << setw(width) << "range" << setw(width)
      << "skew" << setw(width) << "kurt" << setw(width) << "SE" << endl;

  // Lambda function to print out the results.
  auto PrintStatResults = [&](size_t dim, bool rowMajor)
  {
    arma::rowvec feature;
    if (rowMajor)
      feature = ConvTo<arma::rowvec>::From(data.col(dim));
    else
      feature = data.row(dim);

    // f at the front of the variable names means "feature".
    const double fMax = max(feature);
    const double fMin = min(feature);
    const double fMean = arma::mean(feature);
    const double fStd = arma::stddev(feature, population);

    // Print statistics of the given dimension.
    Log::Info << setprecision(precision) << setw(width) << dim <<
        setw(width) << arma::var(feature, population) <<
        setw(width) << fMean <<
        setw(width) << fStd <<
        setw(width) << arma::median(feature) <<
        setw(width) << fMin <<
        setw(width) << fMax <<
        setw(width) << (fMax - fMin) <<
        setw(width) << Skewness(feature, fStd, fMean, population) <<
        setw(width) << Kurtosis(feature, fStd, fMean, population) <<
        setw(width) << StandardError(feature.n_elem, fStd) << endl;
  };

  // If the user specified dimension, describe statistics of the given
  // dimension. If a dimension is not specified, describe all dimensions.
  if (params.Has("dimension"))
  {
    PrintStatResults(dimension, rowMajor);
  }
  else
  {
    const size_t dimensions = rowMajor ? data.n_cols : data.n_rows;
    for (size_t i = 0; i < dimensions; ++i)
    {
      PrintStatResults(i, rowMajor);
    }
  }
  timers.Stop("statistics");
}
