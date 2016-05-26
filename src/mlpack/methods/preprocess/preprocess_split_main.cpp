/**
 * @file preprocess_split_main.cpp
 * @author Keon Woo Kim
 *
 * split data CLI executable
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

PROGRAM_INFO("Split into Train and Test Data", "This "
    "utility takes data and labels and split into a training "
    "set and a test set.");

// Define parameters for data
PARAM_STRING_REQ("input_file", "File containing data,", "i");
PARAM_STRING_REQ("output_train_data", "File name to save train data", "d");
PARAM_STRING_REQ("output_test_data", "File name to save test data", "D");

// Define parameters for labels
PARAM_STRING_REQ("input_label", "File containing labels", "I");
PARAM_STRING_REQ("output_train_label", "File name to save train label", "l");
PARAM_STRING_REQ("output_test_label", "File name to save test label", "L");

// Define optional test ratio, default is 0.2 (Test 20% Train 80%)
PARAM_DOUBLE("test_ratio", "Ratio of test set, defaults to 0.2"
    "if not set", "r", 0.2);

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

  // container for input data and labels
  arma::mat data;
  arma::Mat<size_t> labels;

  // Load Data and Labels
  data::Load(inputFile, data, true);
  data::Load(inputLabel, labels, true);
  arma::Row<size_t> labels_row = labels.row(0); // extract first row

  // Split Data
  const auto value = data::TrainTestSplit(data, labels_row, testRatio);
  Log::Info << "Train Data Count: " << get<0>(value).n_cols << endl;
  Log::Info << "Test Data Count: " << get<1>(value).n_cols << endl;
  Log::Info << "Train Label Count: " << get<2>(value).n_cols << endl;
  Log::Info << "Test Label Count: " << get<3>(value).n_cols << endl;

  // Save Train Data
  data::Save(outputTrainData, get<0>(value), false);

  // Save Test Data
  data::Save(outputTestData, get<1>(value), false);

  // Save Train Label
  data::Save(outputTrainLabel, get<2>(value), false);

  // Save Test Label
  data::Save(outputTestLabel, get<3>(value), false);
}

