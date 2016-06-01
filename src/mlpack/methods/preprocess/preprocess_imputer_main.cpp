/**
 * @file preprocess_imputer_main.cpp
 * @author Keon Kim
 *
 * a utility that provides imputation strategies fore
 * missing values.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/imputer.hpp>

PROGRAM_INFO("Imputer", "This "
    "utility takes an any type of data and provides "
    "imputation strategies for missing data.");

PARAM_STRING_REQ("input_file", "File containing data,", "i");
PARAM_STRING("missing_value", "User defined missing value", "m", "")
PARAM_INT("feature", "the feature to be analyzed", "f", 0);
PARAM_STRING("output_file", "File to save output", "o", "");

using namespace mlpack;
using namespace arma;
using namespace std;

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");
  const string missingValue = CLI::GetParam<string>("missing_value");
  const string outputFile = CLI::GetParam<string>("output_file");
  const size_t featureNumber = (size_t) CLI::GetParam<int>("feature");

  arma::mat data;
  data::DatasetInfo info;

  data::Load(inputFile, data, info,  true, false);
  Log::Debug << "<before change>" << endl;
  Log::Info << data << endl;

  Log::Info << "dataset info: " << endl;
  for (size_t i = 0; i < data.n_rows; ++i)
  {
    Log::Info << info.NumMappings(i) << " mappings in dimension "
        << i << "." << endl;
  }

  Log::Info << "Loading feature: " << featureNumber << endl;
  data::Imputer(data, info, missingValue, featureNumber);

  Log::Debug << "<after change>" << endl;
  Log::Info << data << endl;


  if (!outputFile.empty())
  {
    Log::Info << "Saving model to '" << outputFile << "'." << endl;
    data::Save(outputFile, data, false);
  }
}

