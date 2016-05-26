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
PARAM_STRING_REQ("input_label", "File containing labels", "I");
PARAM_STRING_REQ("training_file", "File name to save train data", "t");
PARAM_STRING_REQ("test_file", "File name to save test data", "T");
PARAM_STRING_REQ("training_labels_file", "File name to save train label", "l");
PARAM_STRING_REQ("test_labels_file", "File name to save test label", "L");

// Define optional test ratio, default is 0.2 (Test 20% Train 80%)
PARAM_DOUBLE("test_ratio", "Ratio of test set, if not set,"
    "the ratio defaults to 0.2", "r", 0.2);

using namespace mlpack;
using namespace arma;
using namespace std;

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");
  const string inputLabel = CLI::GetParam<string>("input_label");
  const string trainingFile = CLI::GetParam<string>("training_file");
  const string testFile = CLI::GetParam<string>("test_file");
  const string trainingLabelsFile = CLI::GetParam<string>("training_labels_file");
  const string testLabelsFile = CLI::GetParam<string>("test_labels_file");
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

  data::Save(trainingFile, get<0>(value), false);
  data::Save(testFile, get<1>(value), false);
  data::Save(trainingLabelsFile, get<2>(value), false);
  data::Save(testLabelsFile, get<3>(value), false);
}

