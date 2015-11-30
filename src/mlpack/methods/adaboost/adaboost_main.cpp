/*
 * @file: adaboost_main.cpp
 * @author: Udit Saxena
 *
 * Implementation of the AdaBoost main file
 *
 *  @code
 *  @article{Schapire:1999:IBA:337859.337870,
 *  author = {Schapire, Robert E. and Singer, Yoram},
 *  title = {Improved Boosting Algorithms Using Confidence-rated Predictions},
 *  journal = {Mach. Learn.},
 *  issue_date = {Dec. 1999},
 *  volume = {37},
 *  number = {3},
 *  month = dec,
 *  year = {1999},
 *  issn = {0885-6125},
 *  pages = {297--336},
 *  numpages = {40},
 *  url = {http://dx.doi.org/10.1023/A:1007614523901},
 *  doi = {10.1023/A:1007614523901},
 *  acmid = {337870},
 *  publisher = {Kluwer Academic Publishers},
 *  address = {Hingham, MA, USA},
 *  keywords = {boosting algorithms, decision trees, multiclass classification,
 *  output coding
 *  }
 *  @endcode
 *
 */

#include <mlpack/core.hpp>
#include "adaboost.hpp"

using namespace mlpack;
using namespace std;
using namespace arma;
using namespace mlpack::adaboost;

PROGRAM_INFO("AdaBoost","This program implements the AdaBoost (or Adaptive Boost)"
 " algorithm. The variant of AdaBoost implemented here is AdaBoost.mh. It uses a"
 " weak learner, either of Decision Stumps or a Perceptron, and over many"
 " iterations, creates a strong learner. It runs these iterations till a tolerance"
 " value is crossed for change in the value of rt."
 "\n"
 "This program allows training of a adaboost object, and then application of "
 "the strong learner to a test dataset.  To train "
 "a training dataset must be passed to --train_file (-t).  Labels can either"
 " be present as the last dimension of the training dataset, or given "
 "explicitly with the --labels_file (-l) parameter.\n"
 "\n"
 "A test file is given through the --test_file (-T) parameter.  The "
 "predicted labels for the test set will be stored in the file specified by "
 "the --output_file (-o) parameter.");

//necessary parameters
PARAM_STRING_REQ("train_file", "A file containing the training set.", "t");
PARAM_STRING_REQ("labels_file", "A file containing labels for the training set.",
  "l");
PARAM_STRING_REQ("test_file", "A file containing the test set.", "T");

//optional parameters.
PARAM_STRING("output", "The file in which the predicted labels for the test set"
    " will be written.", "o", "output.csv");
PARAM_INT("iterations","The maximum number of boosting iterations "
  "to be run", "i", 1000);
PARAM_DOUBLE("tolerance","The tolerance for change in values of rt","e",1e-10);

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);

  const string trainingDataFilename = CLI::GetParam<string>("train_file");
  mat trainingData;
  data::Load(trainingDataFilename, trainingData, true);

  const string labelsFilename = CLI::GetParam<string>("labels_file");
  // Load labels.
  mat labelsIn;
  // data::Load(labelsFilename, labelsIn, true);

  if (CLI::HasParam("labels_file"))
  {
    const string labelsFilename = CLI::GetParam<string>("labels_file");
    // Load labels.
    data::Load(labelsFilename, labelsIn, true);

    // Do the labels need to be transposed?
    if (labelsIn.n_rows == 1)
      labelsIn = labelsIn.t();
  }
  else
  {
    // Extract the labels as the last
    Log::Info << "Using the last dimension of training set as labels." << endl;

    labelsIn = trainingData.row(trainingData.n_rows - 1).t();
    trainingData.shed_row(trainingData.n_rows - 1);
  }

  // helpers for normalizing the labels
  Col<size_t> labels;
  vec mappings;

  // Do the labels need to be transposed?
  if (labelsIn.n_rows == 1)
    labelsIn = labelsIn.t();

  // normalize the labels
  data::NormalizeLabels(labelsIn.unsafe_col(0), labels, mappings);

  const string testingDataFilename = CLI::GetParam<string>("test_file");
  mat testingData;
  data::Load(testingDataFilename, testingData, true);

  const double tolerance = CLI::GetParam<double>("tolerance");

  if (testingData.n_rows != trainingData.n_rows)
    Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
        << "must be the same as training data (" << trainingData.n_rows - 1
        << ")!" << std::endl;
  size_t iterations = (size_t) CLI::GetParam<int>("iterations");

  // define your own weak learner, perceptron in this case.
  // defining the number of iterations of the perceptron.
  size_t iter = 400;

  perceptron::Perceptron<> p(trainingData, labels.t(), max(labels) + 1, iter);

  Timer::Start("Training");
  AdaBoost<> a(trainingData, labels.t(), p, iterations, tolerance);
  Timer::Stop("Training");

  Row<size_t> predictedLabels(testingData.n_cols);
  Timer::Start("testing");
  a.Classify(testingData, predictedLabels);
  Timer::Stop("testing");

  vec results;
  data::RevertLabels(predictedLabels.t(), mappings, results);

  // Save the predicted labels in a transposed form as output.
  const string outputFilename = CLI::GetParam<string>("output_file");
  data::Save(outputFilename, results, true, false);
  return 0;
}
