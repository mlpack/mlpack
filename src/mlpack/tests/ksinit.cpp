/*
	@file ksinit.cpp
  @author Praveen Ch
	

  Tests the working of Kathirvalavakumar Subavathi Initialization for a 
  Feed forward neural network.
*/


#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/softsign_function.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp>

#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/multiclass_classification_layer.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace std;

BOOST_AUTO_TEST_SUITE(KSInitialization);

/**
 * Train and evaluate a vanilla network with the specified initialisation 
   procedure.
 */
template<
		typename PerformanceFunction,
		typename OutputLayerType,
		typename PerformanceFunctionType,
		typename MatType = arma::mat
>


void BuildVanillaNetwork(MatType& trainData,
						 MatType& trainLabels,
						 MatType& testData,
						 MatType& testLabels,
						 const size_t hiddenLayerSize,
						 const size_t maxEpochs,
						 double& trainError,
						 double& testError)
{
	/*
	@param trainError mean squared error of predictions on training data.
	@param testError  mean squared error of predictions on test data.
	 * Construct a feed forward network with trainData.n_rows input nodes,
	 * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
	 * network structure looks like:
	 *
	 *  Input         Hidden        Output
	 *  Layer         Layer         Layer
	 * +-----+       +-----+       +-----+
	 * |     |       |     |       |     |
	 * |     +------>|     +------>|     |
	 * |     |     +>|     |     +>|     |
	 * +-----+     | +--+--+     | +-----+
	 *             |             |
	 *  Bias       |  Bias       |
	 *  Layer      |  Layer      |
	 * +-----+     | +-----+     |
	 * |     |     | |     |     |
	 * |     +-----+ |     +-----+
	 * |     |       |     |
	 * +-----+       +-----+
	 */

	LinearLayer<> inputLayer(trainData.n_rows, hiddenLayerSize);
	BiasLayer<> inputBiasLayer(hiddenLayerSize);
	BaseLayer<PerformanceFunction> inputBaseLayer;

	LinearLayer<> hiddenLayer1(hiddenLayerSize, trainLabels.n_rows);
	BiasLayer<> hiddenBiasLayer1(trainLabels.n_rows);
	BaseLayer<PerformanceFunction> outputLayer;

	OutputLayerType classOutputLayer;

	auto modules = std::tie(inputLayer, inputBiasLayer, inputBaseLayer,
							hiddenLayer1, hiddenBiasLayer1, outputLayer);


    //4.59 is a constant used in the paper.
	KathirvalavakumarSubavathiInitialization init(trainData, 4.59); 

	FFN<decltype(modules), decltype(classOutputLayer), 
      KathirvalavakumarSubavathiInitialization,
			PerformanceFunctionType> net(modules, classOutputLayer, init);

	RMSprop<decltype(net)> opt(net, 0.01, 0.88, 1e-8,
			maxEpochs * trainData.n_cols, 1e-18);

	net.Train(trainData, trainLabels, opt);

	MatType prediction;
 
	// Calculating the mean squared error on the training data.
	net.Predict(trainData, prediction);
	arma::mat squarederror = arma::square(prediction * 1.0 - trainLabels);
	trainError = arma::sum(arma::sum(squarederror)) / trainData.n_cols;

	// Calculating the mean squared error on the test data
	net.Predict(testData, prediction);
	squarederror = arma::square(prediction * 1.0 - testLabels);
	testError = arma::sum(arma::sum(squarederror)) / testData.n_cols;

	
}

/* Error is a structure which has the MSE of Training data and MSE of 
   Validation data respectively. This structure is returned to the parent test 
   case by CrossValidation function.

*/
struct Error
{
	double trainError;
	double validationError;
};

/* CrossValidation function runs a k-fold cross validation on the training data
   by dividing the training data into k equal disjoint subsets. The model is 
   trained on k-1 of these subsets and 1 subset is used as validation data.

   This process is repeated k times assigning each subset to be the validation
   data at most once.

*/

Error CrossValidation(arma::mat& trainData, arma::mat& trainLabels, size_t k,
                      const size_t hiddenLayerSize, const size_t maxEpochs)
{

/*
  @params trainData The data available for training.
  @params trainLabels The labels corresponding to the training data.
  @params k The parameter k in k-fold cross validation.


  @params validationDataSize Number of datapoints in each subset in K-fold CV.

  @params validationTrainData The collection of the k-1 subsets to be used 
                              in training in a particular iteration.

  @params validationTrainLabels The labels corresponding to training data.

  @params validationTestData The data subset which is used as validation data 
                             in a particular iteration.

  @params validationTestLabels The labels corresponding to the validation data.

*/
	size_t validationDataSize = (int) trainData.n_cols / k;

	double validationErrorAvg;
	double trainErrorAvg;

	for (size_t i=0; i < trainData.n_cols; i = i + validationDataSize)
	{
		validationDataSize = (int) trainData.n_cols/k;
		
		arma::mat validationTrainData(trainData.n_rows, trainData.n_cols);
		arma::mat validationTrainLabels(trainLabels.n_rows, trainLabels.n_cols);
		arma::mat validationTestData(trainData.n_rows, validationDataSize);
		arma::mat validationTestLabels(trainLabels.n_rows, validationDataSize);

		if (i + validationDataSize > trainData.n_cols)
			validationDataSize = trainData.n_cols - i;

		validationTestData = 
        trainData.submat(0, i, trainData.n_rows - 1, i + validationDataSize - 1);
		
   	validationTestLabels = 
        trainLabels.submat(0, i, trainLabels.n_rows - 1, 
                           i + validationDataSize - 1);

		validationTrainData = trainData;
		validationTrainData.shed_cols(i, i + validationDataSize - 1);
		

		validationTrainLabels = trainLabels;
		validationTrainLabels.shed_cols(i, i + validationDataSize - 1);
		
		double trainError, validationError;

		BuildVanillaNetwork<SoftsignFunction, 
				MulticlassClassificationLayer,
				MeanSquaredErrorFunction>
      	(validationTrainData, validationTrainLabels, 
         validationTestData, validationTestLabels, hiddenLayerSize, maxEpochs, 
         trainError, validationError);

		trainErrorAvg += trainError;
		validationErrorAvg +=	validationError;

	}

	trainErrorAvg /= k;
	validationErrorAvg  /= k;

	Error e;

	e.trainError = trainErrorAvg;
	e.validationError = validationErrorAvg;

	return e;

}

/*
  AvgCrossValidation function takes a dataset and runs CrossValidation "iter"
  number of times and then return the average training and validation error.
  It shuffles the dataset in every iteration.

*/

Error AvgCrossValidation(arma::mat& dataset, size_t numLabels, size_t iter,
                         const size_t hiddenLayerSize, const size_t maxEpochs)
{

/*
  @params dataset The dataset inclusive of the labels. Assuming the last 
                  "numLabels" number of rows are the labels which the model
                  has to predict. 
  @params numLabels number of rows which are the output labels in the dataset.

  @params iter The number of times Cross Validation has to be run.

  @params hiddenLayerSize The number of nodes in the hidden layer.

  @params maxEpochs The maximum number of epochs for the training.

  @params avgError Returns the train and validation error averaged over "iter"
                   number of Cross Validation results.

*/
  Error avgError = {0.0, 0.0};

  for (int i = 0; i < iter;)
  {
    dataset = arma::shuffle(dataset, 1);

    arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 1 - numLabels, 
                                         dataset.n_cols - 1);
    arma::mat trainLabels = dataset.submat(dataset.n_rows - numLabels, 0, 
                                           dataset.n_rows - 1, 
                                           dataset.n_cols - 1); 
    
    Error e = CrossValidation(trainData, trainLabels, 10, hiddenLayerSize, 
                              maxEpochs);

    // The check is to prevent nan values from affecting the result.
    if (arma::is_finite(e.trainError) && arma::is_finite(e.validationError))
    {
      avgError.trainError += e.trainError;
      avgError.validationError += e.validationError;
      ++i;
    }
    
  }

  avgError.trainError /= iter;
  avgError.validationError /= iter;

  return avgError;
}


/*Test case for the Iris Dataset */

BOOST_AUTO_TEST_CASE(IrisDataset)
{
	arma::arma_rng::set_seed_random();

	double trainErrorThreshold = 0.0008;
	double validationErrorThreshold = 0.0008;

	arma::mat dataset, trainData, trainLabels;

	data::Load("iris.csv", dataset, true);

  dataset /= 10; // Normalization used in the paper.

	Error avgError = AvgCrossValidation(dataset, 1, 10, 3, 52); //Runs CV 10 times

	BOOST_REQUIRE_LE(avgError.trainError, trainErrorThreshold);
	BOOST_REQUIRE_LE(avgError.validationError, validationErrorThreshold);

}

BOOST_AUTO_TEST_CASE(NonLinearFunctionApproximation)
{
	arma::arma_rng::set_seed_random();

	double trainErrorThreshold = 0.0035;
	double validationErrorThreshold = 0.0035;

	arma::mat dataset(11, 500, arma::fill::randu);

  // Eqn 13.1 as given in the paper
	dataset.row(8) = dataset.row(0) % dataset.row(1);

  for (int i = 2; i <= 6; i = i + 2)
    dataset.row(8) += dataset.row(i) % dataset.row(i+1);

  dataset.row(8) /= 4;

  // Eqn 13.2
	dataset.row(9) = dataset.row(0);
  
  for (int i = 1; i <= 7; ++i)
    dataset.row(9) += dataset.row(i);

  dataset.row(9) /= 8;

  // Eqn 13.3
  dataset.row(10) = arma::sqrt(1 - dataset.row(0));
	

  Error avgError = AvgCrossValidation(dataset, 3, 10, 10, 500);
	

	BOOST_REQUIRE_LE(avgError.trainError, trainErrorThreshold);
	BOOST_REQUIRE_LE(avgError.validationError, validationErrorThreshold);

}

BOOST_AUTO_TEST_SUITE_END();

