/*! @page sample_ml_app Sample C++ ML App

@section sample_intro Introduction

This tutorial will help you create a sample machine learning app using mlpack/C++. Although this app
does not cover all the mlpack capabilities, it will walkthrough several APIs to understand how
everything connects. This Windows sample app is created using Visual Studio, but you can easily
adapt it to a different platform by following the provided source code.

@note Before starting, make sure you have built mlpack for Windows following this @ref build_windows "Windows guide"

@section sample_create_project Creating the VS project

- Open Visual Studio and create a new project (Windows Console Application)
- For this sample, the project is named “mymlpackapp” and is located at "C:\myprojects\"

@section sample_project_config Project Configuration

There are different ways in which you can configure your project to link with dependencies. This configuration
is for x64 Debug Mode. If you need Release Mode, please change the paths accordingly (assuming you have built
mlpack and dependencies in Release Mode).

- Right click on the project and select Properties, select the x64 Debug profile
- Under C/C++ > General > Additional Include Directories add:
 - "C:\boost\boost_1_66_0"
 - "C:\mlpack\armadillo-8.500.1\include"
 - "C:\mlpack\mlpack-3.0.2\build\include"
- Under Linker > Input > Additional Dependencies add:
 - "C:\mlpack\mlpack-3.0.2\build\Debug\mlpack.lib"
 - "C:\boost\boost_1_66_0\lib64-msvc-14.1\libboost_serialization-vc141-mt-gd-x64-1_66.lib"
 - "C:\boost\boost_1_66_0\lib64-msvc-14.1\libboost_program_options-vc141-mt-gd-x64-1_66.lib"
- Under Build Events > Post-Build Event > Command Line add:
 - xcopy /y "C:\mlpack\mlpack-3.0.2\build\Debug\mlpack.dll" $(OutDir)
 - xcopy /y "C:\mlpack\mlpack-3.0.2\packages\OpenBLAS.0.2.14.1\lib\native\bin\x64\*.dll" $(OutDir)

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

First step is about loading the dataset. Here we load a CSV dataset, assuming the labels
don't require normalization.

@code
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
printf("\nTraining Accuracy: %f", (double(correct) / double(labels.n_elem)));
@endcode

@section sample_crossvalidation Cross-Validating

To evaluate the classifier, we use K-Fold cross-validation. We also define which metric to use in order
to assess the quality of the trained model.

@code
const size_t k = 10;
KFoldCV<RandomForest<GiniGain, RandomDimensionSelect>, Accuracy> cv(k, 
	dataset, labels, numClasses);
double cvAcc = cv.Evaluate(numTrees, minimumLeafSize);
printf("\nKFoldCV Accuracy: %f", cvAcc);
@endcode

To compute other relevant metrics, such as Precision, Recall and F1:

@code
double cvPrecision = Precision<Binary>::Evaluate(rf, dataset, labels);
printf("\nPrecision: %f", cvPrecision);

double cvRecall = Recall<Binary>::Evaluate(rf, dataset, labels);
printf("\nRecall: %f", cvRecall);

double cvF1 = F1<Binary>::Evaluate(rf, dataset, labels);
printf("\nF1: %f", cvF1);
@endcode

@section sample_save_model Saving the model

Now that our model is trained and validated, we save it to a file so we can use it later.

@code
mlpack::data::Save("mymodel.xml", "model", rf, false,
		mlpack::data::format::xml);
@endcode

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
mat sample("2 12 2 13 1 2 2 1 3 24 3 1 1 1 1 1 0 1 0 1 0 0 0");
mat probabilities;
rf.Classify(sample, predictions, probabilities);
u64 result = predictions.at(0);
printf("\nClassification result: %i (Probabilities: %f/%f)", result,
	probabilities.at(0), probabilities.at(1));
@endcode

@section sample_app_conclussion Final thoughts

Building real-life applications and services using machine learning can be challenging. Hopefully, this
tutorial provides a good starting point that covers the entire workflow you may need to follow while
developing it. You can take a look at the entire source code in the provided sample project.

*/
