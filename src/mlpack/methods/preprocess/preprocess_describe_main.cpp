/**
 * @file preprocess_describe_main.cpp
 * @author Keon Kim
 *
 * Descriptive Statistics Class and CLI executable.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

using namespace mlpack;
using namespace mlpack::data;
using namespace std;
using namespace boost;

PROGRAM_INFO("Descriptive Statistics", "This utility takes a dataset and "
    "prints out the descriptive statistics of the data. Descriptive statistics "
    "is the discipline of quantitatively describing the main features of a "
    "collection of information, or the quantitative description itself. The "
    "program does not modify the original file, but instead prints out the "
    "statistics to the console. The printed result will look like a table."
    "\n\n"
    "Optionally, width and precision of the output can be adjusted by a user "
    "using the --width (-w) and --precision (-p). A user can also select a "
    "specific dimension to analyize if he or she has too many dimensions."
    "--population (-P) is a flag which can be used when the user wants the "
    "dataset to be considered as a population. Otherwise, the dataset will "
    "be considered as a sample."
    "\n\n"
    "So, a simple example where we want to print out statistical facts about "
    "dataset.csv, and keep the default settings, we could run"
    "\n\n"
    "$ mlpack_preprocess_describe -i dataset.csv -v"
    "\n\n"
    "If we want to customize the width to 10 and precision to 5 and consider "
    "the dataset as a population, we could run"
    "\n\n"
    "$ mlpack_preprocess_describe -i dataset.csv -w 10 -p 5 -P -v");

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
  return arma::sum(arma::pow(input - fMean, static_cast<double>(n)));
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

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);
  const size_t dimension = static_cast<size_t>(CLI::GetParam<int>("dimension"));
  const size_t precision = static_cast<size_t>(CLI::GetParam<int>("precision"));
  const size_t width = static_cast<size_t>(CLI::GetParam<int>("width"));
  const bool population = CLI::HasParam("population");
  const bool rowMajor = CLI::HasParam("row_major");

  // Load the data.
  arma::mat& data = CLI::GetParam<arma::mat>("input");

  // Generate boost format recipe.
  const string widthPrecision("%-" + to_string(width) + "." +
      to_string(precision));
  const string widthOnly("%-" + to_string(width) + ".");
  string stringFormat = "";
  string numberFormat = "";

  // We are going to print 11 different categories.
  for (size_t i = 0; i < 11; ++i)
  {
    stringFormat += widthOnly + "s";
    numberFormat += widthPrecision + "f";
  }

  Timer::Start("statistics");
  // Print the headers.
  Log::Info << boost::format(stringFormat)
      % "dim" % "var" % "mean" % "std" % "median" % "min" % "max"
      % "range" % "skew" % "kurt" % "SE" << endl;

  // Lambda function to print out the results.
  auto PrintStatResults = [&](size_t dim, bool rowMajor)
  {
    arma::rowvec feature;
    if (rowMajor)
      feature = arma::conv_to<arma::rowvec>::from(data.col(dim));
    else
      feature = data.row(dim);

    // f at the front of the variable names means "feature".
    const double fMax = arma::max(feature);
    const double fMin = arma::min(feature);
    const double fMean = arma::mean(feature);
    const double fStd = arma::stddev(feature, population);

    // Print statistics of the given dimension.
    Log::Info << boost::format(numberFormat)
        % dim
        % arma::var(feature, population)
        % fMean
        % fStd
        % arma::median(feature)
        % fMin
        % fMax
        % (fMax - fMin) // range
        % Skewness(feature, fStd, fMean, population)
        % Kurtosis(feature, fStd, fMean, population)
        % StandardError(feature.n_elem, fStd)
        << endl;
  };

  // If the user specified dimension, describe statistics of the given
  // dimension. If a dimension is not specified, describe all dimensions.
  if (CLI::HasParam("dimension"))
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
  Timer::Stop("statistics");
}

