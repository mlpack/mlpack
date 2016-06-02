/**
 * @file preprocess_split_main.cpp
 * @author Keon Woo Kim
 *
 * split data CLI executable
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

PROGRAM_INFO("Split Data", "This utility takes data and split into a training "
    "set and a test set. Before the split happens, it shuffles the data in "
    "the each feature. Without (--test_ratio) specified, the default "
    "test-to-training ratio is set to 0.2."
    "\n\n"
    "The program does not modify or write on the original file, but instead "
    "makes a seperate files to save the training and test files; you can "
    "specify the file names with (-training_file) and (-test_file). If the "
    "names are not specified, the program automatically names the training "
    "and test file by attaching 'train_' and 'test_' in front of the "
    "original file name"
    "\n\n"
    "Optionally, a label can be also be splited along with the data at the "
    "same time by specifying (--input_lables) option. Splitting label works "
    "the same as splitting the data and you can also specify the names using "
    "(--trainning_labels_file) and (--test_labels_file).");

// Define parameters for data
PARAM_STRING_REQ("input_file", "File containing data,", "i");
// Define optional parameters
PARAM_STRING("input_labels", "File containing labels", "I", "");
PARAM_STRING("training_file", "File name to save train data", "t", "");
PARAM_STRING("test_file", "File name to save test data", "T", "");
PARAM_STRING("training_labels_file", "File name to save train label", "l", "");
PARAM_STRING("test_labels_file", "File name to save test label", "L", "");

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
  const string inputLabels = CLI::GetParam<string>("input_labels");
  string trainingFile = CLI::GetParam<string>("training_file");
  string testFile = CLI::GetParam<string>("test_file");
  string trainingLabelsFile = CLI::GetParam<string>("training_labels_file");
  string testLabelsFile = CLI::GetParam<string>("test_labels_file");
  const double testRatio = CLI::GetParam<double>("test_ratio");

  // check on data parameters
  if (trainingFile.empty())
  {
    trainingFile = "train_" + inputFile;
    Log::Warn << "You did not specify --training_file. "
      << "Training file name is automatically set to: "
      << trainingFile << endl;
  }
  if (testFile.empty())
  {
    testFile = "test_" + inputFile;
    Log::Warn << "You did not specify --test_file. "
      << "Test file name is automatically set to: " << testFile << endl;
  }

  // check on label parameters
  if (!inputLabels.empty())
  {
    if (!CLI::HasParam("training_labels_file"))
    {
      trainingLabelsFile = "train_" + inputLabels;
      Log::Warn << "You did not specify --training_labels_file. "
        << "Training labels file name is automatically set to: "
        << trainingLabelsFile << endl;
    }
    if (!CLI::HasParam("test_labels_file"))
    {
      testLabelsFile = "test_" + inputLabels;
      Log::Warn << "You did not specify --test_labels_file. "
        << "Test labels file name is automatically set to: "
        << testLabelsFile << endl;
    }
  }
  else
  {
    if (CLI::HasParam("training_labels_file")
        || CLI::HasParam("test_labels_file"))
    {
      Log::Fatal << "When specifying --training_labels_file or "
        << "test_labels_file, you must also specify --input_labels. " << endl;
    }
  }

  // check on test_ratio
  if (CLI::HasParam("test_ratio"))
  {
    //sanity check on test_ratio
    if ((testRatio < 0.0) || (testRatio > 1.0))
    {
      Log::Fatal << "Invalid parameter for test_ratio. "
        << "test_ratio must be between 0.0 and 1.0" << endl;
    }
  }
  else // if test_ratio is not set
  {
    Log::Warn << "You did not specify --test_ratio_file. "
      << "Test ratio is automatically set to: 0.2"<< endl;
  }

  // load data
  arma::mat data;
  data::Load(inputFile, data, true);

  // if parameters for labels exist
  if (CLI::HasParam("input_labels"))
  {
    arma::mat labels;
    data::Load(inputLabels, labels, true);
    arma::rowvec labels_row = labels.row(0); // extract first row

    const auto value = data::Split(data, labels_row, testRatio);
    Log::Info << "Train Data Count: " << get<0>(value).n_cols << endl;
    Log::Info << "Test Data Count: " << get<1>(value).n_cols << endl;
    Log::Info << "Train Label Count: " << get<2>(value).n_cols << endl;
    Log::Info << "Test Label Count: " << get<3>(value).n_cols << endl;

    data::Save(trainingFile, get<0>(value), false);
    data::Save(testFile, get<1>(value), false);
    data::Save(trainingLabelsFile, get<2>(value), false);
    data::Save(testLabelsFile, get<3>(value), false);
  }
  else // split without parameters
  {
    const auto value = data::Split(data, testRatio);
    Log::Info << "Train Data Count: " << get<0>(value).n_cols << endl;
    Log::Info << "Test Data Count: " << get<1>(value).n_cols << endl;

    data::Save(trainingFile, get<0>(value), false);
    data::Save(testFile, get<1>(value), false);
  }
}

