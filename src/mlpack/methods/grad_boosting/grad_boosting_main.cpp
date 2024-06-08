/**
 * @file methods/grad_boosting/grad_boosting_main.cpp
 * @author Abhimanyu Dayal
 *
 * A program to implement gradient boosting model. 
 * This file defines the templates for the documentation, 
 * as well as define the BINDING_FUNCTION()
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


// Defining BINDING_NAME.
#undef BINDING_NAME
#define BINDING_NAME grad_boosting

// ### DEFINING HEADERS

// Importing base components required to write mlpack methods.
#include <mlpack/core.hpp>

// Importing checking core utilities (eg. ReportIgnoredParam etc.)
#include <mlpack/core/util/mlpack_main.hpp>

// Importing the BINDING macros.
#include <mlpack/bindings/cli/mlpack_main.hpp>

// Importing grad_boosting model.
#include <grad_boosting_model.hpp>

// Defining namespaces.
using namespace std;
using namespace mlpack;
using namespace mlpack::data;
using namespace mlpack::util;


// ### SPECIFYING BINDING METADATA

// Program Name.
BINDING_USER_NAME("Gradient Boosting");

// Short description.
BINDING_SHORT_DESC (
  "An implementation of Gradient Boosting for classification. " 
  "A gradient boosting model can be trained and saved; or an "
  "existing gradient boosting model can be used for classification on new points."
);

// Long description.
BINDING_LONG_DESC (
  "This program implements Gradient Boosting algorithm. It uses "
  "weak learners (primarily decision stumps), and trains them "
  "sequentially, such that future learners are trained to detect the "
  "errors (or gradients) of previous learners. The results obtained from "
  "all of these learners are subsequently aggregated to give a final result."
  "Weak learners (i.e. models which predict nearly random outcomes) "
  "are used here because they are individually very fast to train and predict."
  "\n\n"
  "This program allows training of a Gradient Boosting model, and then application of"
  " that model to a test dataset. To train a model, a dataset must be passed"
  " with the " + PRINT_PARAM_STRING("training") + " option. Labels can be "
  "given with the " + PRINT_PARAM_STRING("labels") + " option; if no labels "
  "are specified, the labels will be assumed to be the last column of the "
  "input dataset. Alternately, a Gradient Boosting model may be loaded with the " +
  PRINT_PARAM_STRING("input_model") + " option."
  "\n\n"
  "Once a model is trained or loaded, it may be used to provide class "
  "predictions for a given test dataset. A test dataset may be specified "
  "with the " + PRINT_PARAM_STRING("test") + " parameter. The predicted "
  "classes for each point in the test dataset are output to the " +
  PRINT_PARAM_STRING("predictions") + " output parameter. The Gradient Boosting "
  "model itself is output to the " + PRINT_PARAM_STRING("output_model") +
  " output parameter."
  "\n\n"
);

// Example.
BINDING_EXAMPLE(
  "For example, to run Gradient Boosting on an input dataset " +
  PRINT_DATASET("data") + " with labels " + PRINT_DATASET("labels") +
  "storing the trained model in " + PRINT_MODEL("model") + 
  ", one could use the following command: \n\n" +
  PRINT_CALL("Gradient Boosting", "training", "data", "labels", "labels",
      "output_model", "model") + "\n\n"
  "Similarly, an already-trained model in " + PRINT_MODEL("model") + " can"
  " be used to provide class predictions from test data " +
  PRINT_DATASET("test_data") + " and store the output in " +
  PRINT_DATASET("predictions") + " with the following command: "
  "\n\n" +
  PRINT_CALL("Gradient Boosting", "input_model", "model", "test", "test_data",
      "predictions", "predictions"));

// See also...
BINDING_SEE_ALSO("Gradient Boosting on Wikipedia", "https://en.wikipedia.org/wiki/"
  "Gradient_boosting");
BINDING_SEE_ALSO("Greedy Function Approximation: A Gradient Boosting Machine", 
  "https://jerryfriedman.su.domains/ftp/trebst.pdf");
BINDING_SEE_ALSO("Decision Stump", "#decision_stump");
BINDING_SEE_ALSO("mlpack::grad_boosting::GradientBoosting C++ class documentation",
  "@src/mlpack/methods/grad_boosting/grad_boosting.hpp");

// Input for training.
PARAM_MATRIX_IN("training", "Dataset for training Gradient Boosting.", "t");
PARAM_UROW_IN("labels", "Labels for the training set.", "l");

// Classification options.
PARAM_MATRIX_IN("test", "Test dataset.", "T");
PARAM_UROW_OUT("predictions", "Predicted labels for the test set.", "P");
PARAM_MATRIX_OUT("probabilities", "Predicted class probabilities for each "
  "point in the test set.", "p");

// Training parameter options.
PARAM_INT_IN("numLearners", "Number of weak learners to use", "n", 1);

// Loading/saving of a model.
PARAM_MODEL_IN(GradBoostingModel, "input_model", "Input Gradient Boosting model.", "m");
PARAM_MODEL_OUT(GradBoostingModel, "output_model", "Output trained Gradient Boosting model.",
  "M");

// ### DEFINING BINDING FUNCTION

void BINDING_FUNCTION(util::Params& params, util::Timers& timers) 
{
  // ### CHECKING ERRORS/WARNINGS

  // The user cannot specify both a training file and an input model file.
  RequireOnlyOnePassed(params, { "training", "input_model" });

  // --labels can't be specified without --training.
  ReportIgnoredParam(params, {{ "training", false }}, "labels");

  // --predictions can't be specified without --test.
  ReportIgnoredParam(params, {{ "test", false }}, "predictions");

  // Training parameters are ignored if no training file is given.
  ReportIgnoredParam(params, {{ "training", false }}, "numLearners");

  // If the user gave an input model but no test set, issue a warning.
  if (params.Has("input_model")) {
    RequireAtLeastOnePassed(params, { "test" }, false, "no task will be performed");
  }

  // If the user doesn't specify output_model, output or prediction, issue a warning.
  RequireAtLeastOnePassed(params, { "output_model", "output", "predictions" },
  false, "no results will be saved");

  // ### MODEL TRAINING

  // Initiating pointer to model variable.
  GradBoostingModel* m;

  // If training new model.
  if (params.Has("training")) 
  {
    // Load training data.
    arma::mat trainingData = move(params.Get<arma::mat>("training"));

    // Initiate GradBoosting model.
    m = new GradBoostingModel();

    // Load labels.
    arma::Row<size_t> labelsIn;
    if (params.Has("labels")) 
    {
        labelsIn = move(params.Get<arma::Row<size_t>>("labels"));
    }

    // Extract the labels as the last dimension of the training data.
    else 
    {
        Log::Info << "Using the last dimension of training set as labels." << endl;
        labelsIn = ConvTo<arma::Row<size_t>>::From(
          trainingData.row(trainingData.n_rows - 1)
        );
        trainingData.shed_row(trainingData.n_rows - 1);
    }

    // Helpers for normalizing the labels.
    arma::Row<size_t> labels;

    // Normalize the labels.
    data::NormalizeLabels(labelsIn, labels, m->Mappings());

    // Get other training parameters.
    const int numLearners = params.Get<int>("numLearners");

    // Count number of classes in labels.
    const size_t numClasses = m->Mappings().n_elem;
    Log::Info << numClasses << " classes in dataset." << endl;

    // Start training. 
    timers.Start("grad_boosting_training");
    m->Train(trainingData, labels, numClasses, numLearners);
    timers.Stop("grad_boosting_training");
  }

  // We have a specified input model.
  else 
  {
    m = params.Get<GradBoostingModel*>("input_model");
  }

  // Perform classification on test data, if desired.
  if (params.Has("test")) 
  {
    // Load test data.
    arma::mat testingData = move(params.Get<arma::mat>("test"));

    // Ensure test data has required dimensionality.
    if (testingData.n_rows != m->Dimensionality())
      Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
        << "must be the same as the model dimensionality ("
        << m->Dimensionality() << ")!" << endl;

    // Load labels for test data.
    arma::Row<size_t> predictedLabels(testingData.n_cols);
    arma::mat probabilities;

    // Calculate probabilities of each class if desired.
    if (params.Has("probabilities")) {
      timers.Start("grad_boosting_classification");
      m->Classify(testingData, predictedLabels, probabilities);
      timers.Stop("grad_boosting_classification");
    }
    else {
      timers.Start("grad_boosting_classification");
      m->Classify(testingData, predictedLabels);
      timers.Stop("grad_boosting_classification");
    }

    // Load the calculated results into a results vector
    arma::Row<size_t> results;
    data::RevertLabels(predictedLabels, m->Mappings(), results);

    // Save the predicted labels.
    if (params.Has("output"))
      params.Get<arma::Row<size_t>>("output") = results;
    if (params.Has("predictions"))
      params.Get<arma::Row<size_t>>("predictions") = move(results);
    if (params.Has("probabilities"))
      params.Get<arma::mat>("probabilities") = move(probabilities);
  }

  // Output the obtained model into the output_model parameter.
  params.Get<GradBoostingModel*>("output_model") = m;
}