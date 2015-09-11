/*
 * @file: perceptron_main.cpp
 * @author: Udit Saxena
 *
 * This program runs the Simple Perceptron Classifier.
 *
 * Perceptrons are simple single-layer binary classifiers, which solve linearly
 * separable problems with a linear decision boundary.
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
    "using the --max_iterations (-i) parameter, if the data supplied is "
    "linearly separable.  The perceptron is parameterized by a matrix of weight"
    " vectors that denote the numerical weights of the neural network."
    "\n\n"
    "This program allows loading a perceptron from a model (-i) or training a "
    "perceptron given training data (-t), or both those things at once.  In "
    "addition, this program allows classification on a test dataset (-T) and "
    "will save the classification results to the given output file (-o).  The "
    "perceptron model itself may be saved with a file specified using the -m "
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
    " if an existing perceptron model is passed with -i (--input_model).  "
    "However, note that the number of classes and the dimensionality of all "
    "data must match.  So you cannot pass a perceptron model trained on 2 "
    "classes and then re-train with a 4-class dataset.  Similarly, attempting "
    "classification on a 3-dimensional dataset with a perceptron that has been "
    "trained on 8 dimensions will cause an error."
    );

// Training parameters.
PARAM_STRING("training_file", "A file containing the training set.", "t", "");
PARAM_STRING("labels_file", "A file containing labels for the training set.",
  "l", "");
PARAM_INT("max_iterations","The maximum number of iterations the perceptron is "
  "to be run", "M", 1000);

// Model loading/saving.
PARAM_STRING("input_model", "File containing input perceptron model.", "i", "");
PARAM_STRING("output_model", "File to save trained perceptron model to.", "m",
    "");

// Testing/classification parameters.
PARAM_STRING("test_file", "A file containing the test set.", "T", "");
PARAM_STRING("output_file", "The file in which the predicted labels for the "
    "test set will be written.", "o", "output.csv");

// When we save a model, we must also save the class mappings.  So we use this
// auxiliary structure to store both the perceptron and the mapping, and we'll
// save this.
class PerceptronModel
{
 private:
  Perceptron<>& p;
  arma::vec& map;

 public:
  PerceptronModel(Perceptron<>& p, arma::vec& map) : p(p), map(map) { }

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
  const string trainingDataFile = CLI::GetParam<string>("training_file");
  const string labelsFile = CLI::GetParam<string>("labels_file");
  const string inputModelFile = CLI::GetParam<string>("input_model");
  const string testDataFile = CLI::GetParam<string>("test_file");
  const string outputModelFile = CLI::GetParam<string>("output_model");
  const string outputFile = CLI::GetParam<string>("output_file");
  const size_t maxIterations = (size_t) CLI::GetParam<int>("max_iterations");

  // We must either load a model or train a model.
  if (inputModelFile == "" && trainingDataFile == "")
    Log::Fatal << "Either an input model must be specified with --input_model "
        << "or training data must be given (--training_file)!" << endl;

  // If the user isn't going to save the output model or any predictions, we
  // should issue a warning.
  if (outputModelFile == "" && testDataFile == "")
    Log::Warn << "Output will not be saved!  (Neither --test_file nor "
        << "--output_model are specified.)" << endl;

  // Now, load our model, if there is one.
  Perceptron<>* p = NULL;
  arma::vec mappings;
  if (inputModelFile != "")
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
  if (trainingDataFile != "")
  {
    Log::Info << "Training perceptron on dataset '" << trainingDataFile;
    if (labelsFile != "")
      Log::Info << "' with labels in '" << labelsFile << "'";
    else
      Log::Info << "'";
    Log::Info << " for a maximum of " << maxIterations << " iterations."
        << endl;

    mat trainingData;
    data::Load(trainingDataFile, trainingData, true);

    // Load labels.
    mat labelsIn;

    // Did the user pass in labels?
    if (CLI::HasParam("labels_file"))
    {
      // Load labels.
      const string labelsFile = CLI::GetParam<string>("labels_file");
      data::Load(labelsFile, labelsIn, true);
    }
    else
    {
      // Use the last row of the training data as the labels.
      Log::Info << "Using the last dimension of training set as labels."
          << endl;
      labelsIn = trainingData.row(trainingData.n_rows - 1).t();
      trainingData.shed_row(trainingData.n_rows - 1);
    }

    // Do the labels need to be transposed?
    if (labelsIn.n_rows == 1)
      labelsIn = labelsIn.t();

    // Normalize the labels.
    Col<size_t> labels;
    data::NormalizeLabels(labelsIn.unsafe_col(0), labels, mappings);

    // Now, if we haven't already created a perceptron, do it.  Otherwise, make
    // sure the dimensions are right, then continue training.
    if (p == NULL)
    {
      // Create and train the classifier.
      Timer::Start("training");
      p = new Perceptron<>(trainingData, labels.t(), max(labels) + 1,
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
            << " '" << trainingDataFile << "' has " << trainingData.n_rows
            << "dimensions!" << endl;
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
  if (testDataFile != "")
  {
    Log::Info << "Classifying dataset '" << testDataFile << "'." << endl;
    mat testData;
    data::Load(testDataFile, testData, true);

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
    vec results;
    data::RevertLabels(predictedLabels.t(), mappings, results);

    // Save the predictedLabels, but we have to transpose them.
    data::Save(outputFile, results, false /* non-fatal */, false);
  }

  // Lastly, do we need to save the output model?
  if (outputModelFile != "")
  {
    PerceptronModel pm(*p, mappings);
    data::Save(outputModelFile, "perceptron_model", pm);
  }

  // Clean up memory.
  delete p;
}
