/**
 * @file sample_ml_app.hpp
 * @author German Lancioni

@page sample_ml_app Sample C++ ML App for Windows

@section sample_intro Introduction

This tutorial will help you create a sample machine learning app using mlpack/C++. Although this app
does not cover all the mlpack capabilities, it will walkthrough several APIs to understand how
everything connects. This Windows sample app is created using Visual Studio, but you can easily
adapt it to a different platform by following the provided source code.

@note Before starting, make sure you have built mlpack for Windows following this @ref build_windows "Windows guide"

@section sample_create_project Creating the VS project

- Open Visual Studio and create a new project (Windows Console Application)
- For this sample, the project is named “sample-ml-app”

@section sample_project_config Project Configuration

There are different ways in which you can configure your project to link with dependencies. This configuration
is for x64 Debug Mode. If you need Release Mode, please change the paths accordingly (assuming you have built
mlpack and dependencies in Release Mode).

- Right click on the project and select Properties, select the x64 Debug profile
- Under C/C++ > General > Additional Include Directories add:
@code
 - C:\boost\boost_1_71_0\lib\native\include
 - C:\mlpack\armadillo-9.800.3\include
 - C:\mlpack\mlpack-3.4.1\build\include
@endcode
- Under Linker > Input > Additional Dependencies add:
@code
 - C:\mlpack\mlpack-3.4.1\build\Debug\mlpack.lib
 - C:\boost\boost_1_71_0\lib64-msvc-14.2\libboost_serialization-vc142-mt-gd-x64-1_71.lib
@endcode
- Under Build Events > Post-Build Event > Command Line add:
@code
 - xcopy /y "C:\mlpack\mlpack-3.4.1\build\Debug\mlpack.dll" $(OutDir)
 - xcopy /y "C:\mlpack\mlpack-3.4.1\packages\OpenBLAS.0.2.14.1\lib\native\bin\x64\*.dll" $(OutDir)
@endcode

@note Recent versions of Visual Studio set "Conformance Mode" enabled by default. This causes some issues with
the armadillo library. If you encounter this issue, disable "Conformance Mode" under C/C++ > Language.

@section sample_app_goal The app goal

This app aims to exercise an end-to-end machine learning workflow. We will cover:

- Loading and preparing a dataset
- Training (using Random Forest as example)
- Computing the training accuracy
- Cross-Validation using K-Fold
- Metrics gathering (accuracy, precision, recall, F1)
- Saving the trained model to disk
- Loading the model
- Classifying a new sample

@section sample_headers_namespaces Headers and namespaces

For this app, we will need to include the following headers (i.e. add into stdafx.h):

@code
#include "mlpack/core.hpp"
#include "mlpack/methods/random_forest/random_forest.hpp"
#include "mlpack/methods/decision_tree/random_dimension_select.hpp"
#include "mlpack/core/cv/k_fold_cv.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include "mlpack/core/cv/metrics/precision.hpp"
#include "mlpack/core/cv/metrics/recall.hpp"
#include "mlpack/core/cv/metrics/F1.hpp"
@endcode

Also, we will use the following namespaces:

@code
using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::cv;
@endcode

@section sample_load_dataset Loading the dataset

First step is about loading the dataset. Different dataset file formats are supported, but here
we load a CSV dataset, and we assume the labels don't require normalization.

@note Make sure you update the path to your dataset file. For this sample, you can simply
copy "mlpack/tests/data/german.csv" and paste into a new "data" folder in your project directory.

@code
mat dataset;
bool loaded = mlpack::data::Load("data/german.csv", dataset);
if (!loaded)
  return -1;
@endcode

Then we need to extract the labels from the last dimension of the dataset and remove the
labels from the training set:

@code
Row<size_t> labels;
labels = conv_to<Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
dataset.shed_row(dataset.n_rows - 1);
@endcode

We now have our dataset ready for training.

@section sample_training Training

This app will use a Random Forest classifier. At first we define the classifier parameters and then
we create the classifier to train it.

@code
const size_t numClasses = 2;
const size_t minimumLeafSize = 5;
const size_t numTrees = 10;

RandomForest<GiniGain, RandomDimensionSelect> rf;

rf = RandomForest<GiniGain, RandomDimensionSelect>(dataset, labels,
    numClasses, numTrees, minimumLeafSize);
@endcode

Now that the training is completed, we quickly compute the training accuracy:

@code
Row<size_t> predictions;
rf.Classify(dataset, predictions);
const size_t correct = arma::accu(predictions == labels);
cout << "\nTraining Accuracy: " << (double(correct) / double(labels.n_elem));
@endcode

@section sample_crossvalidation Cross-Validating

Instead of training the Random Forest directly, we could also use K-fold cross-validation for training,
which will give us a measure of performance on a held-out test set. This can give us a better estimate 
of how the model will perform when given new data. We also define which metric to use in order
to assess the quality of the trained model.

@code
const size_t k = 10;
KFoldCV<RandomForest<GiniGain, RandomDimensionSelect>, Accuracy> cv(k, 
    dataset, labels, numClasses);
double cvAcc = cv.Evaluate(numTrees, minimumLeafSize);
cout << "\nKFoldCV Accuracy: " << cvAcc;
@endcode

To compute other relevant metrics, such as Precision, Recall and F1:

@code
double cvPrecision = Precision<Binary>::Evaluate(rf, dataset, labels);
cout << "\nPrecision: " << cvPrecision;

double cvRecall = Recall<Binary>::Evaluate(rf, dataset, labels);
cout << "\nRecall: " << cvRecall;

double cvF1 = F1<Binary>::Evaluate(rf, dataset, labels);
cout << "\nF1: " << cvF1;
@endcode

@section sample_save_model Saving the model

Now that our model is trained and validated, we save it to a file so we can use it later. Here we save the
model that was trained using the entire dataset. Alternatively, we could extract the model from the cross-validation
stage by using \c cv.Model()

@code
mlpack::data::Save("mymodel.xml", "model", rf, false);
@endcode

We can also save the model in \c bin format ("mymodel.bin") which would result in a smaller file.

@section sample_load_model Loading the model

In a real-life application, you may want to load a previously trained model to classify new samples.
We load the model from a file using:

@code
mlpack::data::Load("mymodel.xml", "model", rf);
@endcode

@section sample_classify_sample Classifying a new sample

Finally, the ultimate goal is to classify a new sample using the previously trained model. Since the
Random Forest classifier provides both predictions and probabilities, we obtain both.

@code
// Create a test sample containing only one point.  Because Armadillo is
// column-major, this matrix has one column (one point) and the number of rows
// is equal to the dimensionality of the point (23).
mat sample("2; 12; 2; 13; 1; 2; 2; 1; 3; 24; 3; 1; 1; 1; 1; 1; 0; 1; 0; 1;"
    " 0; 0; 0");
mat probabilities;
rf.Classify(sample, predictions, probabilities);
u64 result = predictions.at(0);
cout << "\nClassification result: " << result << " , Probabilities: " <<
    probabilities.at(0) << "/" << probabilities.at(1);
@endcode

@section sample_app_conclussion Final thoughts

Building real-life applications and services using machine learning can be challenging. Hopefully, this
tutorial provides a good starting point that covers the basic workflow you may need to follow while
developing it. You can take a look at the entire source code in the provided sample project located here:
"doc/examples/sample-ml-app".

*/
