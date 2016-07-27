/**
 * @file preprocess_describe_main.cpp
 * @author Keon Kim
 *
 * Descriptive Statistics Class and CLI executable.
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
PARAM_STRING_IN_REQ("input_file", "File containing data,", "i");
PARAM_INT_IN("dimension", "Dimension of the data. Use this to specify a "
    "dimension", "d", 0);
PARAM_INT_IN("precision", "Precision of the output statistics.", "p", 4);
PARAM_INT_IN("width", "Width of the output table.", "w", 8);
PARAM_FLAG("population", "If specified, the program will calculate statistics "
    "assuming the dataset is the population. By default, the program will "
    "assume the dataset as a sample.", "P");

/**
* Calculates the sum of deviations to the Nth Power
*
* @param input Vector that captures a dimension of a dataset
* @param rowMean Mean of the given vector.
* @return sum of nth power deviations
*/
double SumNthPowerDeviations(const arma::rowvec& input,
    const double& rowMean,
    const size_t Nth) // Degree of Power
{
  double sum = 0;
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    sum += pow(input(i) - rowMean, Nth);
  }
  return sum;
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
    const double& rowStd,
    const double& rowMean,
    const bool population)
{
  double skewness = 0;
  double S3 = pow(rowStd, 3);
  double M3 = SumNthPowerDeviations(input, rowMean, 3);
  double n = input.n_elem;
  if (population)
  {
    // Calculate Population Skewness
    skewness = n * M3 / (n * n * S3);
  }
  else
  {
    // Calculate Sample Skewness
    skewness = n * M3 / ((n-1) * (n-2) * S3);
  }
  return skewness;
}
/**
 * Calculates kurtosis of the given vector.
 *
 * @param input Vector that captures a dimension of a dataset
 * @param rowStd Standard Deviation of the given vector.
 * @param rowMean Mean of the given vector.
 * @return Kurtosis of the given vector.
 */
double Kurtosis(const arma::rowvec& input,
    const double& rowStd,
    const double& rowMean,
    const bool population)
{
  double kurtosis = 0;
  double M4 = SumNthPowerDeviations(input, rowMean, 4);
  double n = input.n_elem;
  if (population)
  {
    // Calculate Population Excess Kurtosis
    double M2 = SumNthPowerDeviations(input, rowMean, 2);
    kurtosis = n * (M4 / pow(M2, 2)) - 3;
  }
  else
  {
    // Calculate Sample Excess Kurtosis
    double S4 = pow(rowStd, 4);
    double norm3 = (3 * (n-1) * (n-1)) / ((n-2) * (n-3));
    double normC = (n * (n+1))/((n-1) * (n-2) * (n-3));
    double normM = M4 / S4;
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
double StandardError(const arma::rowvec& input, const double rowStd)
{
   return rowStd / sqrt(input.n_elem);
}

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);
  const string inputFile = CLI::GetParam<string>("input_file");
  const size_t dimension = static_cast<size_t>(CLI::GetParam<int>("dimension"));
  const size_t precision = static_cast<size_t>(CLI::GetParam<int>("precision"));
  const size_t width = static_cast<size_t>(CLI::GetParam<int>("width"));
  const bool population = CLI::HasParam("population");

  // Load the data
  arma::mat data;
  data::Load(inputFile, data, false, true /*transpose*/);

  // Generate boost format recipe.
  const string widthPrecision("%-"+
    to_string(width)+ "." +
    to_string(precision));
  const string widthOnly("%-"+
    to_string(width)+ ".");
  string stringFormat = "";
  string numberFormat = "";
  for (size_t i = 0; i < 11; ++i)
  {
    stringFormat += widthOnly + "s";
    numberFormat += widthPrecision + "f";
  }

  Timer::Start("statistics");
  // Headers
  Log::Info << boost::format(stringFormat)
      % "dim" % "var" % "mean" % "std" % "median" % "min" % "max"
      % "range" % "skew" % "kurt" % "SE" << endl;

  // If the user specified dimension, describe statistics of the given
  // dimension. If it dimension not specified, describe all dimensions.
  if (CLI::HasParam("dimension"))
  {
    // Extract row of the data with the given dimension.
    arma::rowvec row = data.row(dimension);
    // These variables are kept for future calculations.
    double rowMax = arma::max(row);
    double rowMin = arma::min(row);
    double rowMean = arma::mean(row);
    double rowStd = arma::stddev(row, population);

    // Print statistics of the given dimension.
    Log::Info << boost::format(numberFormat)
        % dimension
        % arma::var(row, population)
        % rowMean
        % rowStd
        % arma::median(row)
        % rowMin
        % rowMax
        % (rowMax - rowMin) // range
        % Skewness(row, rowStd, rowMean, population)
        % Kurtosis(row, rowStd, rowMean, population)
        % StandardError(row, rowStd) << endl;
  }
  else
  {
    for (size_t i = 0; i < data.n_rows; ++i)
    {
      // Extract each dimension of the data.
      arma::rowvec row = data.row(i);
      // These variables are kept for future calculations.
      double rowMax = arma::max(row);
      double rowMin = arma::min(row);
      double rowMean = arma::mean(row);
      double rowStd = arma::stddev(row, population);

      // Print statistics of the row i.
      Log::Info << boost::format(numberFormat)
          % i
          % arma::var(row, population)
          % rowMean
          % rowStd
          % arma::median(row)
          % rowMin
          % rowMax
          % (rowMax - rowMin) // range
          % Skewness(row, rowStd, rowMean, population)
          % Kurtosis(row, rowStd, rowMean, population)
          % StandardError(row, rowStd) << endl;
    }
  }
  Timer::Stop("statistics");
}

