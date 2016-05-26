/**
 * @file preprocess_split_data_main.cpp
 * @author Keon Woo Kim
 *
 * split data CLI executable
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

PROGRAM_INFO("Split Test Data", "This "
    "utility takes an any type of data and evaluates the "
    "mean, mode, and etc of an input (--input_file).");

// Define parameters for data
PARAM_STRING_REQ("input_file", "File containing data,", "i");
PARAM_STRING_REQ("output_train_data", "File to save output", "d");
PARAM_STRING_REQ("output_test_data", "File containing test data", "D");

// Define parameters for labels
PARAM_STRING_REQ("input_label", "File containing labels", "I");
PARAM_STRING_REQ("output_train_label", "File containing train Label", "l");
PARAM_STRING_REQ("output_test_label", "File containing test label", "L");

// Define optional test ratio, default is 0.2 (Test 20% Train 80%)
PARAM_DOUBLE("test_ratio", "test ratio", "r", 0.2);

using namespace mlpack;
using namespace arma;
using namespace std;

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  // data
  const string inputFile = CLI::GetParam<string>("input_file");
  const string outputTrainData = CLI::GetParam<string>("output_train_data");
  const string outputTestData = CLI::GetParam<string>("output_test_data");

  // labels
  const string inputLabel = CLI::GetParam<string>("input_label");
  const string outputTrainLabel = CLI::GetParam<string>("output_train_label");
  const string outputTestLabel = CLI::GetParam<string>("output_test_label");

  // Ratio
  const double testRatio = CLI::GetParam<double>("test_ratio");

  // container for input data
  arma::mat data;
  data::DatasetInfo info;
  // container for input labels
  arma::Mat<size_t> labels;

  // Load Data
  data::Load(inputFile, data, info,  true);
  Log::Info << inputFile << ": " << endl;
  Log::Info << data << endl;

  // Load Label
  data::Load(inputLabel, labels, true);
  arma::Row<size_t> labels_row = labels.row(0); // extract first row
  Log::Info << inputLabel << ": " << endl;
  Log::Info << labels_row << endl;

  // Split Data
  const auto value = data::TrainTestSplit(data, labels_row, testRatio);
  Log::Info << "Train Data Count: " << get<0>(value).n_cols << endl;
  Log::Info << "Test Data Count: " << get<1>(value).n_cols << endl;
  Log::Info << "Train Label Count: " << get<2>(value).n_cols << endl;
  Log::Info << "Test Label Count: " << get<3>(value).n_cols << endl;

  Log::Info << "Train Data Sample: " << endl;
  Log::Info << get<0>(value) << endl;

  // Save Train Data
  Log::Info << "Saving train data to '" << outputTrainData << "'." << endl;
  data::Save(outputTrainData, get<0>(value), false);

  // Save Test Data
  Log::Info << "Saving test data to '" << outputTestData << "'." << endl;
  data::Save(outputTestData, get<1>(value), false);

  // Save Train Label
  Log::Info << "Saving train labels to '" << outputTrainLabel << "'." << endl;
  data::Save(outputTrainLabel, get<2>(value), false);

  // Save Test Label
  Log::Info << "Saving test labels to '" << outputTestLabel << "'." << endl;
  data::Save(outputTestLabel, get<3>(value), false);
}

