/**
 * @file methods/naive_bayes/nbc_train_main.cpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * author Dirk Eddelbuettel
 *
 * Implementation of training for a Simple Naive Bayes Classifier.
 *
 * This classifier does parametric naive bayes classification assuming that the
 * features are sampled from a Gaussian distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME nbc_train

#include <mlpack/core/util/mlpack_main.hpp>

#include "naive_bayes_classifier.hpp"
#include "naive_bayes_model.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;
using namespace arma;

// Program Name.
BINDING_USER_NAME("Parametric Naive Bayes Classifier training");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the Naive Bayes Classifier, used for classification. "
    "Given labeled data, an NBC model is be trained for later use "
    "for classification on new data.");

// Long description.
BINDING_LONG_DESC(
    "Implements the Naive Bayes classifier on the given labeled "
    "training set for us of"
    " that trained model to classify the points in a given test set."
    "\n\n"
    "The training set is specified with the " +
    PRINT_PARAM_STRING("training") + " parameter.  Labels may be either the "
    "last row of the training set, or alternately the " +
    PRINT_PARAM_STRING("labels") + " parameter may be specified to pass a "
    "separate matrix of labels."
    "\n\n"
    "The " + PRINT_PARAM_STRING("incremental_variance") + " parameter can be "
    "used to force the training to use an incremental algorithm for calculating"
    " variance.  This is slower, but can help avoid loss of precision in some "
    "cases.");

// Example.
BINDING_EXAMPLE(
    IMPORT_EXT_LIB() + "\n" +
    IMPORT_SPLIT() + "\n" +
    IMPORT_THIS("naive_bayes", "train", "classify", "probabilities") + "\n" +
    GET_DATASET("X", "http://datasets.mlpack.org/iris.csv") + "\n" +
    GET_DATASET("y", "http://datasets.mlpack.org/iris_labels.csv") + "\n" +
    SPLIT_TRAIN_TEST("X", "y", "X_train", "y_train", "X_test", "y_test",
    "0.2") + "\n" +
    CREATE_OBJECT("model", "naive_bayes_trees") + "\n" +
    CALL_METHOD("model", "train", "training", "X_train", "labels", "y_train"));

// See also...
BINDING_SEE_ALSO("@softmax_regression", "#softmax_regression");
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("Naive Bayes classifier on Wikipedia",
    "https://en.wikipedia.org/wiki/Naive_Bayes_classifier");
BINDING_SEE_ALSO("NaiveBayesClassifier C++ class documentation",
    "@doc/user/methods/naive_bayes_classifier.md");

// Model loading/saving.
PARAM_MODEL_OUT(NBCModel, "output_model", "File to save trained "
    "Naive Bayes model to.", "M");

// Training parameters.
PARAM_MATRIX_IN_REQ("training", "A matrix containing the training set.", "t");
PARAM_UROW_IN("labels", "A vector containing labels for the training set.",
    "l");
PARAM_FLAG("incremental_variance", "The variance of each class will be "
    "calculated incrementally.", "I");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Train a model.
  NBCModel* model = new NBCModel();
  mat trainingData = std::move(params.Get<mat>("training"));
  Row<size_t> labels;

  // Did the user pass in labels?
  if (params.Has("labels"))
  {
    // Load labels.
    Row<size_t> rawLabels = std::move(params.Get<Row<size_t>>("labels"));
    NormalizeLabels(rawLabels, labels, model->mappings);
  }
  else
  {
    // Use the last row of the training data as the labels.
    Log::Info << "Using last dimension of training data as training labels."
        << endl;
    NormalizeLabels(trainingData.row(trainingData.n_rows - 1), labels,
        model->mappings);
    // Remove the label row.
    trainingData.shed_row(trainingData.n_rows - 1);
  }
  const bool incrementalVariance = params.Has("incremental_variance");

  timers.Start("nbc_training");
  model->nbc = NaiveBayesClassifier<>(trainingData, labels,
      model->mappings.n_elem, incrementalVariance);
  timers.Stop("nbc_training");

  params.Get<NBCModel*>("output_model") = model;
}
