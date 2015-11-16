#include <mlpack/core.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include <memory>
#include <set>

// Define parameters for the executable.
//PROGRAM_INFO("Softmax Regression", "This program performs softmax regression "
//    "on the given dataset and able to store the learned parameters.");

// Required options.
PARAM_STRING("training_file", "A file containing the training set (the matrix "
    "of predictors, X).", "t", "");
PARAM_STRING("labels_file", "A file containing labels (0 or 1) for the points "
    "in the training set (y). The labels must order as a row", "l", "");

// Model loading/saving.
PARAM_STRING("input_model", "File containing existing model (parameters).", "i",
    "");
PARAM_STRING("output_model", "File to save trained logistic regression model "
    "to.", "m", "");

// Testing.
PARAM_STRING("test_data", "File containing test dataset.", "T", "");
PARAM_STRING("test_labels", "File containing test labels.", "L", "");

// Softmax configuration options.
PARAM_INT("max_iterations", "Maximum number of iterations before "
          "terminates.", "M", 400);

PARAM_INT("number_of_classes", "Number of classes for classification, "
          "if you do not specify, it will measure it out automatic",
          "n", 0);

PARAM_DOUBLE("lambda", "L2-regularization constant", "r", 0.0001);

PARAM_FLAG("intercept", "Add intercept term, if not specify, "
           "the intercept term will not be added", "t");

size_t calculateNumberOfClasses(size_t numClasses,
                                arma::Row<size_t> const &trainLabels);

template<typename Model>
void testPredictAcc(const std::string &testFile,
                    const std::string &testLabels,
                    size_t numClasses,
                    const Model &model);

template<typename Model>
std::unique_ptr<Model> trainSoftmax(const std::string &trainingFile,
                                    const std::string &labelFile,
                                    const std::string &inputModelFile,
                                    size_t maxIterations);

int main(int argc, char** argv)
{
  using namespace mlpack;

  CLI::ParseCommandLine(argc, argv);

  const std::string trainingFile = CLI::GetParam<std::string>("training_file");
  const std::string inputModelFile = CLI::GetParam<std::string>("input_model");

  // One of inputFile and modelFile must be specified.
  if(inputModelFile.empty() && trainingFile.empty())
  {
    Log::Fatal << "One of --input_model or --training_file must be specified."
               << std::endl;
  }

  const std::string labelFile = CLI::GetParam<std::string>("labels_file");
  if(!trainingFile.empty() && labelFile.empty())
  {
    Log::Fatal << "--label_file must be specified with --training_file"
               << std::endl;
  }

  const int maxIterations = CLI::GetParam<int>("max_iterations");

  if (maxIterations < 0)
  {
    Log::Fatal << "Invalid value for maximum iterations (" << maxIterations <<
                  ")! Must be greater than or equal to 0." << std::endl;
  }

  const std::string outputModelFile = CLI::GetParam<std::string>("output_model");

  // Make sure we have an output file if we're not doing the work in-place.
  if (outputModelFile.empty())
  {
    Log::Warn << "--output_model is not set; "
              << "no results will be saved." << std::endl;
  }  


  using SM = regression::SoftmaxRegression<>;
  std::unique_ptr<SM> sm = trainSoftmax<SM>(trainingFile,
                                            labelFile,
                                            inputModelFile,
                                            maxIterations);

  testPredictAcc(CLI::GetParam<std::string>("test_data"),
                 CLI::GetParam<std::string>("test_labels"),
                 sm->NumClasses(), *sm);

  if(!outputModelFile.empty())
  {
    data::Save(CLI::GetParam<std::string>("output_model"),
               "softmax_regression_model", *sm, true);
  }
}

size_t calculateNumberOfClasses(size_t numClasses,
                                arma::Row<size_t> const &trainLabels)
{
  if(numClasses == 0){
    const std::set<size_t> unique_labels(std::begin(trainLabels),
                                         std::end(trainLabels));
    numClasses = unique_labels.size();
  }

  return numClasses;
}

template<typename Model>
void testPredictAcc(const std::string &testFile,
                    const std::string &testLabelsFile,
                    size_t numClasses,
                    const Model &model)
{
    using namespace mlpack;
    if(testFile.empty() && testLabelsFile.empty())
    {
      return;
    }

    if((!testFile.empty() && testLabelsFile.empty()) ||
        (testFile.empty() && !testLabelsFile.empty()))
    {
      Log::Fatal << "--test_file must be specified with --test_labels and vice versa"
                 << std::endl;
    }

    if(!testFile.empty() && !testLabelsFile.empty())
    {
      arma::mat testData;
      arma::Row<size_t> testLabels;
      testData.load(testFile, arma::auto_detect);
      testData = testData.t();
      testLabels.load(testLabelsFile, arma::auto_detect);

      if(testData.n_cols!= testLabels.n_elem)
      {
          Log::Fatal << "Labels of --test_labels should same as the samples size "
                        "of --test_data " << std::endl;
      }

      arma::vec predictLabels;
      model.Predict(testData, predictLabels);
      std::vector<size_t> bingoLabels(numClasses, 0);
      std::vector<size_t> labelSize(numClasses, 0);
      for(arma::uword i = 0; i != predictLabels.n_elem; ++i)
      {
        if(predictLabels(i) == testLabels(i))
        {
          ++bingoLabels[testLabels(i)];
        }
        ++labelSize[testLabels(i)];
      }
      size_t totalBingo = 0;
      for(size_t i = 0; i != bingoLabels.size(); ++i)
      {
        std::cout<<"Accuracy of label "<<i<<" is "
                 <<(bingoLabels[i]/static_cast<double>(labelSize[i]))
                 <<std::endl;
        totalBingo += bingoLabels[i];
      }
      std::cout<<"\nTotal accuracy is "
               <<(totalBingo)/static_cast<double>(predictLabels.n_elem)
               <<std::endl;
    }
}

template<typename Model>
std::unique_ptr<Model> trainSoftmax(const std::string &trainingFile,
                                    const std::string &labelFile,
                                    const std::string &inputModelFile,
                                    size_t maxIterations)
{
    using namespace mlpack;

    using SRF = regression::SoftmaxRegressionFunction;
    using SM = regression::SoftmaxRegression<>;

    std::unique_ptr<Model> sm;
    if(!inputModelFile.empty())
    {

      sm.reset(new Model(0, 0, false));
      mlpack::data::Load(inputModelFile,
                         "softmax_regression_model",
                         *sm, true);
    }
    else
    {
      arma::mat trainData;
      arma::Row<size_t> trainLabels;
      trainData.load(trainingFile, arma::auto_detect);
      trainData = trainData.t();
      trainLabels.load(labelFile, arma::auto_detect);

      //load functions of mlpack do not works on windows, it will complain
      //"[FATAL] Unable to detect type of 'softmax_data.txt'; incorrect extension?"
      //data::Load(inputFile, trainData, true);
      //data::Load(labelFile, trainLabels, true);

      if(trainData.n_cols != trainLabels.n_elem)
      {
        Log::Fatal << "Samples of input_data should same as the size "
                      "of input_label " << std::endl;
      }

      //size_t numClasses = CLI::GetParam<int>("number_of_classes");
      const size_t numClasses =
                calculateNumberOfClasses(CLI::GetParam<int>("number_of_classes"),
                                         trainLabels);

      const bool intercept = !CLI::HasParam("intercept") ? false : true;

      SRF smFunction(trainData, trainLabels, numClasses,
                     intercept, CLI::GetParam<double>("lambda"));
      const size_t numBasis = 5;
      optimization::L_BFGS<SRF> optimizer(smFunction, numBasis, maxIterations);
      sm.reset( new Model(optimizer));
    }

    return sm;
}
