#include <mlpack/core.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include <set>

// Define parameters for the executable.
PROGRAM_INFO("Softmax Regression", "This program performs softmax regression "
    "on the given dataset and able to store the learned parameters.");

// Required options.
PARAM_STRING_REQ("input_data", "Input dataset to perform training on(read from files).", "i");
PARAM_STRING_REQ("input_label",
                 "Input labels to perform training"
                 " on(read from files). The labels must order as a row", "l");

// Output options.
PARAM_STRING("output_file", "If specified, the trained results will write into this "
             "file; Else the training results will not be saved", "p", "");

// Softmax configuration options.
PARAM_INT("max_iterations", "Maximum number of iterations before "
          "terminates.", "m", 400);

PARAM_INT("number_of_classes", "Number of classes for classification, "
          "if you do not specify, it will measure it out automatic",
          "n", 0);

PARAM_DOUBLE("lambda", "L2-regularization constant", "r", 0.0001);

PARAM_FLAG("intercept", "Add intercept term, if not specify, "
           "the intercept term will not be added", "t");


int main(int argc, char** argv)
{
  using namespace mlpack;

  CLI::ParseCommandLine(argc, argv);

  const auto inputFile = CLI::GetParam<std::string>("input_data");
  const auto labelFile = CLI::GetParam<std::string>("input_label");
  const auto maxIterations = CLI::GetParam<int>("max_iterations");

  if (maxIterations < 0)
  {
    Log::Fatal << "Invalid value for maximum iterations (" << maxIterations <<
                  ")! Must be greater than or equal to 0." << std::endl;
  }

  // Make sure we have an output file if we're not doing the work in-place.
  if (!CLI::HasParam("output_file"))
  {
    Log::Warn << "--output_file is not set; "
              << "no results will be saved." << std::endl;
  }

  arma::mat trainData;
  arma::Row<size_t> trainLabels;
  trainData.load(inputFile, arma::auto_detect);
  trainData = trainData.t();
  trainLabels.load(labelFile, arma::auto_detect);

  //load functions of mlpack do not works on windows, it will complain
  //"[FATAL] Unable to detect type of 'softmax_data.txt'; incorrect extension?"
  //data::Load(inputFile, trainData, true);
  //data::Load(labelFile, trainLabels, true);

  std::cout<<trainData<<"\n\n";
  std::cout<<trainLabels<<"\n\n";
  if(trainData.n_cols != trainLabels.n_elem)
  {
    Log::Fatal << "Samples of input_data should same as the size "
                  "of input_label " << std::endl;
  }

  size_t numClasses = CLI::GetParam<int>("number_of_classes");
  if(numClasses == 0){
    const std::set<size_t> unique_labels(std::begin(trainLabels),
                                         std::end(trainLabels));
    numClasses = unique_labels.size();
  }

  const auto intercept = !CLI::HasParam("intercept") ? false : true;

  using SRF = regression::SoftmaxRegressionFunction;
  SRF smFunction(trainData, trainLabels, numClasses,
               intercept, CLI::GetParam<double>("lambda"));

  const size_t numBasis = 5;
  optimization::L_BFGS<SRF> optimizer(smFunction, numBasis, maxIterations);
  regression::SoftmaxRegression<> sm(optimizer);

  if(CLI::HasParam("output_file"))
  {
    data::Save(CLI::GetParam<std::string>("output_file"),
               "softmax_regression", sm, true);
  }
}
