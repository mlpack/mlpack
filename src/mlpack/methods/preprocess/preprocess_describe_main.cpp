/**
 * @file preprocess_describe_main.cpp
 * @author Keon Kim
 *
 * Descriptive Statistics Class and CLI executable.
 */
#include <mlpack/core.hpp>

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

PROGRAM_INFO("Descriptive Statistics", "This utility takes a dataset prints "
    "out the statistical facts about the data.");

// Define parameters for data.
PARAM_STRING_REQ("input_file", "File containing data,", "i");
PARAM_INT("dimension", "Dimension of the data", "d", 0);
PARAM_INT("precision", "preferred precision of the result", "p", 2);

// Statistics class, it calculates most of the statistical elements in its
// constructor.
template <typename T>
class Statistics
{
 public:
  Statistics(arma::Mat<T> input, size_t norm_type = 1, bool columnMajor = true):
      data(input)
  {
    minVec = arma::min(data, columnMajor);
    maxVec = arma::max(data, columnMajor);
    meanVec = arma::mean(data, columnMajor);
    medianVec = arma::median(data, columnMajor);
    stdVec = arma::stddev(data, 1, columnMajor);
    varVec = arma::var(data, 1, columnMajor);
  }
  double Min(const size_t dimension) const
  {
    return minVec(dimension);
  }

  double Max(const size_t dimension) const
  {
    return maxVec(dimension);
  }

  double Range(const size_t dimension) const
  {
    return maxVec(dimension) - minVec(dimension);
  }

  double Mean(const size_t dimension) const
  {
    return meanVec(dimension);
  }

  double Median(const size_t dimension) const
  {
    return medianVec(dimension);
  }

  double Variance(const size_t dimension) const
  {
    return varVec(dimension);
  }

  double StandardDeviation(const size_t dimension) const
  {
    return stdVec(dimension);
  }

  double Skewness(const size_t dimension) const
  {
    return this->CentralMoment(3, dimension);
  }

  double Kurtosis(const size_t dimension) const
  {
    return this->CentralMoment(4, dimension);
  }

  double RawMoment(const size_t order, const size_t dimension) const
  {
    // E(x)^order
    double moment = 0;
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      moment += pow(data(dimension, i), order);
    }
    return moment / data.n_cols;
  }

  double CentralMoment(const size_t order, const size_t dimension) const
  {
    // E(X-u)^order
    if (order == 1)
    {
      return 0.0;
    }
    double moment = 0;
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      moment += pow(data(dimension, i) - meanVec(dimension), order);
    }
    return moment / data.n_cols;
  }

  double StandardError(const size_t dimension) const
  {
     return stdVec(dimension) / sqrt(data.n_cols);
  }
 private:
  arma::Mat<T> data;

  arma::vec minVec;
  arma::vec maxVec;
  arma::vec meanVec;
  arma::vec medianVec;
  arma::vec stdVec;
  arma::vec varVec;
};

/**
 * Make sure a CSV is loaded correctly.
 */
int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);
  const string inputFile = CLI::GetParam<string>("input_file");
  const size_t dimension = (size_t) CLI::GetParam<int>("dimension");
  const size_t precision = (size_t) CLI::GetParam<int>("precision");

  // Load the data
  arma::mat data;
  data::Load(inputFile, data);

  Statistics<double> stats(data);

  // Headers
  Log::Info << left << setw(8) << "dim";
  Log::Info << left << setw(8) << "var";
  Log::Info << left << setw(8) << "mean";
  Log::Info << left << setw(8) << "std";
  Log::Info << left << setw(8) << "median";
  Log::Info << left << setw(8) << "min";
  Log::Info << left << setw(8) << "max";
  Log::Info << left << setw(8) << "range";
  Log::Info << left << setw(10) << "skewness";
  Log::Info << left << setw(10) << "kurtosis";
  Log::Info << left << setw(10) << "SE";
  Log::Info << endl;

  // If the user specified dimension, describe statistics of the given
  // dimension. If it dimension not specified, describe all dimensions.
  if (CLI::HasParam("dimension"))
  {
      // Options
      Log::Info << fixed;
      Log::Info << setprecision(2);
      // Describe Data
      Log::Info << left << setw(8) << dimension;
      Log::Info << left << setw(8) << stats.Variance(dimension);
      Log::Info << left << setw(8) << stats.Mean(dimension);
      Log::Info << left << setw(8) << stats.StandardDeviation(dimension);
      Log::Info << left << setw(8) << stats.Median(dimension);
      Log::Info << left << setw(8) << stats.Min(dimension);
      Log::Info << left << setw(8) << stats.Max(dimension);
      Log::Info << left << setw(8) << stats.Range(dimension);
      Log::Info << left << setw(10) << stats.Skewness(dimension);
      Log::Info << left << setw(10) << stats.Kurtosis(dimension);
      Log::Info << left << setw(10) << stats.StandardError(dimension);
      Log::Info << endl;
  }
  else
  {
    for (size_t i = 0; i < data.n_rows; ++i)
    {
      // Options
      Log::Info << fixed;
      Log::Info << setprecision(2);
      // Describe Data
      Log::Info << left << setw(8) << i;
      Log::Info << left << setw(8) << stats.Variance(i);
      Log::Info << left << setw(8) << stats.Mean(i);
      Log::Info << left << setw(8) << stats.StandardDeviation(i);
      Log::Info << left << setw(8) << stats.Median(i);
      Log::Info << left << setw(8) << stats.Min(i);
      Log::Info << left << setw(8) << stats.Max(i);
      Log::Info << left << setw(8) << stats.Range(i);
      Log::Info << left << setw(10) << stats.Skewness(i);
      Log::Info << left << setw(10) << stats.Kurtosis(i);
      Log::Info << left << setw(10) << stats.StandardError(i);
      Log::Info << endl;
    }
  }
}

