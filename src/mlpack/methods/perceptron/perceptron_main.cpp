/*
 * @file: perceptron_main.cpp
 * @author: Udit Saxena
 *
 * This program runs the Simple Perceptron Classifier.
 *
 * Perceptrons are simple single-layer binary classifiers, which solve linearly
 * separable problems with a linear decision boundary.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include "perceptron.hpp"

using namespace mlpack;
using namespace mlpack::perceptron;
using namespace std;
using namespace arma;

PROGRAM_INFO("Perceptron",
    "This program implements a perceptron, which is a single level neural "
    "network. The perceptron makes its predictions based on a linear predictor "
    "function combining a set of weights with the feature vector.  The "
    "perceptron learning rule is able to converge, given enough iterations "
    "using the --max_iterations (-n) parameter, if the data supplied is "
    "linearly separable.  The perceptron is parameterized by a matrix of weight"
    " vectors that denote the numerical weights of the neural network."
    "\n\n"
    "This program allows loading a perceptron from a model (-m) or training a "
    "perceptron given training data (-t), or both those things at once.  In "
    "addition, this program allows classification on a test dataset (-T) and "
    "will save the classification results to the given output file (-o).  The "
    "perceptron model itself may be saved with a file specified using the -M "
    "option."
    "\n\n"
    "The training data given with the -t option should have class labels as its"
    " last dimension (so, if the training data is in CSV format, labels should "
    "be the last column).  Alternately, the -l (--labels_file) option may be "
    "used to specify a separate file of labels."
    "\n\n"
    "All these options make it easy to train a perceptron, and then re-use that"
    " perceptron for later classification.  The invocation below trains a "
    "perceptron on 'training_data.csv' (and 'training_labels.csv)' and saves "
    "the model to 'perceptron.xml'."
    "\n\n"
    "$ perceptron -t training_data.csv -l training_labels.csv -m perceptron.csv"
    "\n\n"
    "Then, this model can be re-used for classification on 'test_data.csv'.  "
    "The example below does precisely that, saving the predicted classes to "
    "'predictions.csv'."
    "\n\n"
    "$ perceptron -i perceptron.xml -T test_data.csv -o predictions.csv"
    "\n\n"
    "Note that all of the options may be specified at once: predictions may be "
    "calculated right after training a model, and model training can occur even"
    " if an existing perceptron model is passed with -m (--input_model_file).  "
    "However, note that the number of classes and the dimensionality of all "
    "data must match.  So you cannot pass a perceptron model trained on 2 "
    "classes and then re-train with a 4-class dataset.  Similarly, attempting "
    "classification on a 3-dimensional dataset with a perceptron that has been "
    "trained on 8 dimensions will cause an error."
    );

// Training parameters.
PARAM_MATRIX_IN("training", "A matrix containing the training set.", "t");
PARAM_UMATRIX_IN("labels", "A matrix containing labels for the training set.",
    "l");
PARAM_INT_IN("max_iterations","The maximum number of iterations the perceptron "
    "is to be run", "n", 1000);

// Model loading/saving.
PARAM_STRING_IN("input_model_file", "File containing input perceptron model.",
    "m", "");
PARAM_STRING_OUT("output_model_file", "File to save trained perceptron model "
    "to.", "M");

// Testing/classification parameters.
PARAM_MATRIX_IN("test", "A matrix containing the test set.", "T");
PARAM_UMATRIX_OUT("output", "The matrix in which the predicted labels for the"
    " test set will be written.", "o");

// When we save a model, we must also save the class mappings.  So we use this
// auxiliary structure to store both the perceptron and the mapping, and we'll
// save this.
class PerceptronModel
{
 private:
  Perceptron<>& p;
  Col<size_t>& map;

 public:
  PerceptronModel(Perceptron<>& p, Col<size_t>& map) : p(p), map(map) { }

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(p, "perceptron");
    ar & data::CreateNVP(map, "mappings");
  }
};

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // First, get all parameters and validate them.
  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string outputModelFile = CLI::GetParam<string>("output_model_file");
  const size_t maxIterations = (size_t) CLI::GetParam<int>("max_iterations");

  // We must either load a model or train a model.
  if (!CLI::HasParam("input_model_file") && !CLI::HasParam("training"))
    Log::Fatal << "Either an input model must be specified with "
        << "--input_model_file or training data must be given "
        << "(--training_file)!" << endl;

  // If the user isn't going to save the output model or any predictions, we
  // should issue a warning.
  if (!CLI::HasParam("output_model_file") && !CLI::HasParam("test"))
    Log::Warn << "Output will not be saved!  (Neither --test_file nor "
        << "--output_model_file are specified.)" << endl;

  if (!CLI::HasParam("test") && CLI::HasParam("output"))
    Log::Warn << "--output_file will be ignored because --test_file is not "
        << "specified." << endl;

  if (CLI::HasParam("test") && !CLI::HasParam("output"))
    Log::Warn << "--output_file not specified, so the predictions for "
        << "--test_file will not be saved." << endl;

  // Now, load our model, if there is one.
  Perceptron<>* p = NULL;
  Col<size_t> mappings;
  if (CLI::HasParam("input_model_file"))
  {
    Log::Info << "Loading saved perceptron from model file '" << inputModelFile
        << "'." << endl;

    // The parameters here are invalid, but we are about to load the model
    // anyway...
    p = new Perceptron<>(0, 0);
    PerceptronModel pm(*p, mappings); // Also load class mappings.
    data::Load(inputModelFile, "perceptron_model", pm, true);
  }

  // Next, load the training data and labels (if they have been given).
  if (CLI::HasParam("training"))
  {
    Log::Info << "Training perceptron on dataset '"
        << CLI::GetUnmappedParam<mat>("training");
    if (CLI::HasParam("labels"))
      Log::Info << "' with labels in '"
          << CLI::GetUnmappedParam<Mat<size_t>>("labels") << "'";
    else
      Log::Info << "'";
    Log::Info << " for a maximum of " << maxIterations << " iterations."
        << endl;

    mat trainingData = std::move(CLI::GetParam<mat>("training"));

    // Load labels.
    Mat<size_t> labelsIn;

    // Did the user pass in labels?
    if (CLI::HasParam("labels"))
    {
      labelsIn = std::move(CLI::GetParam<arma::Mat<size_t>>("labels"));
    }
    else
    {
      // Use the last row of the training data as the labels.
      Log::Info << "Using the last dimension of training set as labels."
          << endl;
      labelsIn = arma::conv_to<arma::Mat<size_t>>::from(
          trainingData.row(trainingData.n_rows - 1).t());
      trainingData.shed_row(trainingData.n_rows - 1);
    }

    // Do the labels need to be transposed?
    if (labelsIn.n_cols == 1)
      labelsIn = labelsIn.t();

    // Normalize the labels.
    Row<size_t> labels;
    data::NormalizeLabels(labelsIn.row(0), labels, mappings);

    // Now, if we haven't already created a perceptron, do it.  Otherwise, make
    // sure the dimensions are right, then continue training.
    if (p == NULL)
    {
      // Create and train the classifier.
      Timer::Start("training");
      p = new Perceptron<>(trainingData, labels, max(labels) + 1,
          maxIterations);
      Timer::Stop("training");
    }
    else
    {
      // Check dimensionality.
      if (p->Weights().n_rows != trainingData.n_rows)
      {
        Log::Fatal << "Perceptron from '" << inputModelFile << "' is built on "
            << "data with " << p->Weights().n_rows << " dimensions, but data in"
            << " '" << CLI::GetUnmappedParam<arma::mat>("training") << "' has "
            << trainingData.n_rows << "dimensions!" << endl;
      }

      // Check the number of labels.
      if (max(labels) + 1 > p->Weights().n_cols)
      {
        Log::Fatal << "Perceptron from '" << inputModelFile << "' has "
            << p->Weights().n_cols << " classes, but the training data has "
            << max(labels) + 1 << " classes!" << endl;
      }

      // Now train.
      Timer::Start("training");
      p->MaxIterations() = maxIterations;
      p->Train(trainingData, labels.t());
      Timer::Stop("training");
    }
  }

  // Now, the training procedure is complete.  Do we have any test data?
  if (CLI::HasParam("test"))
  {
    Log::Info << "Classifying dataset '"
        << CLI::GetUnmappedParam<arma::mat>("test") << "'." << endl;
    mat testData = std::move(CLI::GetParam<arma::mat>("test"));

    if (testData.n_rows != p->Weights().n_rows)
    {
      Log::Fatal << "Test data dimensionality (" << testData.n_rows << ") must "
          << "be the same as the dimensionality of the perceptron ("
          << p->Weights().n_rows << ")!" << endl;
    }

    // Time the running of the perceptron classifier.
    Row<size_t> predictedLabels(testData.n_cols);
    Timer::Start("testing");
    p->Classify(testData, predictedLabels);
    Timer::Stop("testing");

    // Un-normalize labels to prepare output.
    Row<size_t> results;
    data::RevertLabels(predictedLabels, mappings, results);

    // Save the predicted labels.
    if (CLI::HasParam("output"))
      CLI::GetParam<arma::Mat<size_t>>("output") = std::move(results);
  }

  // Lastly, do we need to save the output model?
  if (CLI::HasParam("output_model_file"))
  {
    PerceptronModel pm(*p, mappings);
    data::Save(outputModelFile, "perceptron_model", pm);
  }

  // Clean up memory.
  delete p;
}
