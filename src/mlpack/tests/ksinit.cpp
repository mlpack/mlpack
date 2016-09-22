/*
	@file ksinit.cpp
  @author Praveen Ch
	

  Tests the working of Kathirvalavakumar Subavathi Initialization for a 
  Feed forward neural network.
*/


#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/identity_function.hpp>

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
	arma::mat squarederror = arma::square(prediction*1.0 - trainLabels);
	trainError = arma::sum(arma::sum(squarederror)) / trainData.n_cols;

	// Calculating the mean squared error on the test data
	net.Predict(testData, prediction);
	squarederror = arma::square(prediction*1.0 - testLabels);
	testError = arma::sum(arma::sum(squarederror)) / testData.n_cols;

	
}

/* Error is a structure which has the MSE of Training data and MSE of 
   Validation data respectively. This structure is returned to the parent test 
   case by CrossValidation function.

*/
struct Error
{
	double trainErrorAvg;
	long double validationErrorAvg;
};

/* CrossValidation function runs a k-fold cross validation on the training data
   by dividing the training data into k equal disjoint subsets. The model is 
   trained on k-1 of these subsets and 1 subset is used as validation data.

   This process is repeated k times assigning each subset to be the validation
   data at most once.

*/

Error CrossValidation(arma::mat& trainData, arma::mat& trainLabels, size_t k)
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
	size_t validationDataSize = (int) trainData.n_cols/k;

	long double validationErrorAvg;
	double trainErrorAvg;

	for (size_t i=0; i < trainData.n_cols; i=i+validationDataSize)
	{
		validationDataSize = (int) trainData.n_cols/k;
		
		arma::mat validationTrainData(trainData.n_rows, trainData.n_cols);
		arma::mat validationTrainLabels(trainLabels.n_rows, trainLabels.n_cols);
		arma::mat validationTestData(trainData.n_rows, validationDataSize);
		arma::mat validationTestLabels(trainLabels.n_rows, validationDataSize);

		if (i + validationDataSize > trainData.n_cols)
			validationDataSize = trainData.n_cols - i;

		validationTestData = 
        trainData.submat(0,i,trainData.n_rows-1, i+validationDataSize-1);
		
    validationTestLabels = 
        trainLabels.submat(0, i, trainLabels.n_rows-1, i+validationDataSize-1);

		validationTrainData = trainData;
		validationTrainData.shed_cols(i, i+validationDataSize-1);
		

		validationTrainLabels = trainLabels;
		validationTrainLabels.shed_cols(i, i+validationDataSize-1);
		
		double trainError, validationError;

		BuildVanillaNetwork<IdentityFunction,
											MulticlassClassificationLayer,
											MeanSquaredErrorFunction>
      	(validationTrainData, validationTrainLabels, 
         validationTestData, validationTestLabels, 3, 52, 
         trainError, validationError);

		trainErrorAvg += trainError;
		validationErrorAvg +=	validationError;

	}

	trainErrorAvg /= k;
	validationErrorAvg  /= k;

	Error E;

	E.trainErrorAvg = trainErrorAvg;
	E.validationErrorAvg = validationErrorAvg;

	return E;

}


/*Test case for the Iris Dataset */

BOOST_AUTO_TEST_CASE(IrisDataset)
{
	double trainErrorThreshold = 0.0007;
	double validationErrorThreshold = 0.0007;

	arma::mat dataset, trainData, trainLabels;

	data::Load("iris.csv", dataset, true);

	arma::mat bias(1, dataset.n_cols, arma::fill::ones);

	dataset.insert_rows(0, bias);

	dataset /= 10; // Normalization used in the paper.

	trainData = dataset.submat(0, 0, dataset.n_rows-2, dataset.n_cols-1);
	trainLabels = 
      dataset.submat(dataset.n_rows-1, 0, dataset.n_rows-1, dataset.n_cols-1);


	Error E = CrossValidation(trainData, trainLabels, 10);

	BOOST_REQUIRE_LE(E.trainErrorAvg, trainErrorThreshold);
	BOOST_REQUIRE_LE(E.validationErrorAvg, validationErrorThreshold);

}


BOOST_AUTO_TEST_SUITE_END();

