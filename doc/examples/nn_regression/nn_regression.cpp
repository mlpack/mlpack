/**
 * @file main.cpp
 * @author Zoltan Somogyi
 *
 * \brief MLPACK TUTORIAL: neural network regression
 * \details Real world example which shows how to create a neural network mlpack/C++ model for regression,
 * how to save and load the model and then use it for prediction (inference).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/adam/adam_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/prereqs.hpp>

using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

//TODO(zso): this should go into mlpack/core/data/feature_normalization.hpp
namespace mlpack {
	namespace data {

		/** 
		* \class FeatureNormalization
		* \brief Data normalization extension module for mlpack.
		* 
		* \details Normalization of all features in the input data into the same range provides in most of the cases
	    * a much better accuracy and smaller and faster models! Subsequent normalizations of new data for the same model
		* must use the same min, max vectors and therefore they are saved in the FeatureNormalization class! You can also
		* serialize the FeatureNormalization object and reconstruct the min, max vectors e.g. for later inference/prediction.
		*
		* @author Zoltan Somogyi
		*/
		class FeatureNormalization
		{
		public:
			/** Constructor */
			FeatureNormalization(double scalelower, double scaleupper, arma::vec & _itemMin, arma::vec & _itemMax)
				: scale_upper(scaleupper), scale_lower(scalelower), itemMin(&_itemMin), itemMax(&_itemMax) {}

			/**
			* \fn Normalize
			*
			* Normalize the data into the range provided by the scale_lower and scale_upper parameters 
			* by using the itemMin and itemMax vectors or if they do not exist yet then calculate 
			* itemMin and itemMax first for the input data.
			* The operation is in place (modifies the input data).
			*
			* @param data Input dataset
			* @see arma::mat in mlpack documentation
			*/
			void Normalize(arma::mat& data)
			{
				// Use the itemMin, itemMax vectors if provided (e.g. for subsequent normalization with the same range)
				// or make them if not provided
				if (itemMin->size() < 1 || itemMax->size() < 1) {
					*itemMin = arma::vec(data.n_rows, arma::fill::zeros);
					*itemMax = arma::vec(data.n_rows, arma::fill::zeros);
					for (int i = 0; i < data.n_rows; i++)
					{
						(*itemMin)(i) = arma::min(data.row(i));
						(*itemMax)(i) = arma::max(data.row(i));
					}
				}

				data.each_col([&](arma::vec& datapoint)
				{
					for (int i = 0; i < datapoint.size(); i++)
					{
						// Scale the data into the appropriate range given by scale_upper and scale_lower. 
						// Each column (feature) in the data file has a minimum and maximum value which 
						// are used to scale the data.
						double Dx = scale_upper - scale_lower;
						double val = datapoint(i);
						// skip single-valued attribute
						if ((*itemMin)(i) != (*itemMax)(i)) {
							if (val == (*itemMin)(i))
								val = scale_lower;
							else if (val == (*itemMax)(i))
								val = scale_upper;
							else
								val = scale_lower + Dx * (val - (*itemMin)(i)) / ((*itemMax)(i) - (*itemMin)(i));
						}
						datapoint(i) = val;
					}
				});
			}

			/**
			* \fn Denormalize
			* 
			* Denormalize value.
			*
			* @param item Feature item to denormalize (zero based index of the column in the original dataset)
			* @param value The value before denormalization
			* @return the denormalized value
			*/
			double Denormalize(const size_t item, const double value) const
			{
				double Dx = scale_upper - scale_lower;
				double val = (*itemMin)(item) + (value - scale_lower) * ((*itemMax)(item) - (*itemMin)(item)) / Dx;
				return val;
			}

			/** @return Min vector*/
			const arma::vec& Min() const { return *itemMin; }
			/** @return Max vector*/
			const arma::vec& Max() const { return *itemMax; }
			/** @return ScaleUpper parameter*/
			const double ScaleUpper() const { return scale_upper; }
			/** @return ScaleLower parameter*/
			const double ScaleLower() const { return scale_lower; }

			/**
			* Serialization of the itemMin, itemMax vectors and scaling properties.
			*/
			template<typename Archive>
			void serialize(Archive& ar, const unsigned int /* version */)
			{
				ar & BOOST_SERIALIZATION_NVP(*itemMin);
				ar & BOOST_SERIALIZATION_NVP(*itemMax);
				ar & BOOST_SERIALIZATION_NVP(scale_upper);
				ar & BOOST_SERIALIZATION_NVP(scale_lower);
			}

		private:
			arma::vec* itemMin = NULL;
			arma::vec* itemMax = NULL;
			double scale_upper;
			double scale_lower;
		};

	} // namespace data
} // namespace mlpack



/**
 * The main program
 * How to use: set the appropriate parameters hereunder and compile
 * @author Zoltan Somogyi
 */
int main()
{
	//--------------------------------------------------------------------------------------------------------
	//! SET THE FOLLOWING PARAMETERS:

	//!	- data_path: the path to the data directory
	std::string data_path("../../");		

	//! \note mlpack expects TSV extension for tab delimited and CSV extension for comma delimited files!

	//! - data_file: is the data file name.
	std::string data_file("bodyfat.tsv");
	// NOTE: You could add another data file here; for this example we just use the training data file.
	//! - test_data_file: is the test data file name.
	std::string test_data_file("bodyfat.tsv");
	//! - iColDecision: is the decision column in the dataset indexed from 0!
	int iColDecision = 0;

	/** \note In this example we use a very small dataset, therefore splitting it will not give
	*		  an optimal accuracy (MSQE) for the test cases! You should use more data if splitting!
	*/

	//! - RATIO: The dataset is randomly split into training and validation parts with this ratio.
	constexpr double RATIO = 0.01; //1%
	
	//! \note Determining the right number of neurons/nodes is a trial and error process.

	//! - H1: The number of neurons in the 1st layer.
	constexpr int H1 = 200;
	//! - H2: The number of neurons in the 2nd layer.
	constexpr int H2 = 150;
	//! - H3: The number of neurons in the 3rd layer.
	constexpr int H3 = 80;	

	/** \note The used SGD/AdamUpdate optimizer is very fast, therefore a high number of iterations are ok!
	 *		    With other optimizers (e.g. Sarah+) you may have to decrease this number significantly!
	 */

	 //! - ITERATIONS_PER_CYCLE: Number of iteration per cycle.
	constexpr int ITERATIONS_PER_CYCLE = 10000;
	//! - CYCLES: Number of cycles.
	constexpr int CYCLES = 80;
	//! - STEP_SIZE: Step size of the optimizer.
	constexpr double STEP_SIZE = 5e-5;
	//! - BATCH_SIZE: Number of data points in each iteration of SGD. Power of 2 is better for data parallelism.
	constexpr int BATCH_SIZE = 10;
	//! - STOP_TOLERANCE: Stop tolerance; a very small number means that we do not stop but do all iterations
	constexpr int STOP_TOLERANCE = 1e-5;

	// SET THE FOLLOWING PARAMETERS (END) --------------------------------------------------------------------

	/** 
		\brief ABOUT THE DATASET bodyfat.tsv (freely available dataset)

		\details 
		The bodyfat dataset contains estimates of the percentage of body fat determined by underwater weighing 
		and various body circumference measurements for 252 men. Accurate measurement of body fat is very expensive, 
		but by using machine learning it is possible to calculate a prediction with good accuracy by just using
		some low cost measurements of the body. The columns in the dataset are the following:

			Percent body fat (%) => this is the decision column (what we want to get from the model).
			Age (years)
			Weight (lbs)
			Height (inches)
			Neck circumference (cm)
			Chest circumference (cm)
			Abdomen 2 circumference (cm)
			Hip circumference (cm)
			Thigh circumference (cm)
			Knee circumference (cm)
			Ankle circumference (cm)
			Biceps (extended) circumference (cm)
			Forearm circumference (cm)
			Wrist circumference (cm)
			Density determined from underwater weighing
	*/

	std::cout << "\n[SAMPLE:BEGIN]\n";

	// Load the dataset
	std::cout << "Loading " << data_file << " dataset...\n";
	
	mat dataset;
	// TSV is loaded transposed (mlpack: columns are samples, rows are dimensions/features)
	bool loaded = mlpack::data::Load(data_path + data_file, dataset);
	if (!loaded)
		return -1;

	// Splitting the dataset on training and validation parts.
	mat train, valid;
	data::Split(dataset, train, valid, RATIO);

	// Extract the decision column before removing it in the next step
	/** \note - Rows and columns are switched in mlpack! The input data file contains the features in columns which are
	 *		  transferred to rows in mlpack. The records (rows) in the data file become columns in mlpack.
	 */
	mat trainY = train.row(iColDecision);
	mat validY = valid.row(iColDecision);

	// Getting the training and validating datasets with features only.
	// We take the full range of data and then remove the decision column (row in armadillo) in the next step.
	mat trainX = train.submat(0, 0, train.n_rows - 1, train.n_cols - 1);
	mat validX = valid.submat(0, 0, valid.n_rows - 1, valid.n_cols - 1);
	// Remove the decision column (row in armadillo storage system!)
	trainX.shed_row(iColDecision);
	validX.shed_row(iColDecision);

	// Normalize the data into range[0,1]
	/** \note - Normalization of all features in the input data into the same range provides in most of the cases
	 *		  a much better accuracy and smaller and faster models!
	 */
	arma::vec itemMin;
	arma::vec itemMax;
	mlpack::data::FeatureNormalization dn(0.0, 1.0, itemMin, itemMax);
	dn.Normalize(trainX);
	/** \note - Any subsequent normalization of data should use the itemMin and itemMax vectors from the first normalization!
	 *		  itemMin and itemMax are calculated in mlpack::data::FeatureNormalization!
	 */
	dn.Normalize(validX);
	
	/** 
	 * \note - You can check how the data is normalized by saving it as follows:
	 * \code
	 * trainX.quiet_save(data_path + "trainX.csv", arma::file_type::csv_ascii); 
	 * validX.quiet_save(data_path + "validX.csv", arma::file_type::csv_ascii);
	 * \endcode
	 */

	/**
	 * \note - MeanSquaredError is the output layer that is used for regression.
	 */
	// Specifying the NN model. RandomInitialization means that initial weights in neurons 
	// are generated randomly in the interval from -1 to 1.
	FFN <MeanSquaredError<>, RandomInitialization> model;

	/** 
	* \note - There are a lot of layer types available in mlpack. There are activation layers and
	* there are connection layers. Activation layers are for example TanHLayer, ReLULayer, PReLU, SoftPlusLayer, 
	* SigmoidLayer, etc. Connection layer is for example the Linear layer. There are also some special layers
	* as for example DropOut, BatchNorm, LayerNorm, etc. Read the mlpack documentation for more info!
	*/

	// This intermediate layer is needed for connection between input
	// data and the next TanHLayer layer. Parameters specify the number of input features
	// and number of neurons in the next layer.
	// Connaction layer intput => :
	model.Add<Linear<> >(trainX.n_rows, H1);
	// Activation layer:
	model.Add<TanHLayer<> >();
	// Connection layer between two activation layers
	model.Add<Linear<> >(H1, H2);
	// Activation layer
	model.Add<TanHLayer<> >();
	// Connection layer
	model.Add<Linear<> >(H2, H3);
	// Activation layer
	model.Add<TanHLayer<> >();
	// Connection layer => output
	// The output of one neuron is the regression output for one record.
	model.Add<Linear<> >(H3, 1);

	std::cout << "Training the AI model...\n";

	// Setting parameters Stochastic Gradient Descent (SGD) optimizer with Adam update.
	SGD <AdamUpdate> optimizer(
		// Step size of the optimizer.
		STEP_SIZE,
		// Batch size. Number of data points that are used in each iteration.
		BATCH_SIZE,
		// Max number of iterations
		ITERATIONS_PER_CYCLE,
		// Tolerance, used as a stopping condition.
		STOP_TOLERANCE,
		// Shuffle. If optimizer should take random data points from the dataset at each iteration.
		true,
		// Adam update policy.
		AdamUpdate(1e-8, 0.9, 0.999));

	// Cycles for monitoring the process of a solution.
	for (int i = 0; i <= CYCLES; i++)
	{
		// Train neural network. If this is the first iteration, weights are
		// random, using current values as starting point otherwise.
		model.Train(trainX, trainY, optimizer);

		// Don't reset optimizer's parameters between cycles.  
		optimizer.ResetPolicy() = false;
		
		// Getting predictions on training data points. 
		mat predOut;
		model.Predict(trainX, predOut);

		// Calculate mean square error on the prediction
		double trainAccuracy = arma::mean(arma::mean(arma::square(predOut - trainY)));
		
		// Getting predictions on validating data points.
		model.Predict(validX, predOut);

		// Calculate mean square error on the prediction
		double validAccuracy = arma::mean(arma::mean(arma::square(predOut - validY)));

		// Write the results to the screen at each cycle (mean square error)
		std::cout << i << " - accuracy: train = " << trainAccuracy << " (MSQE)," << " valid = " << validAccuracy << " (MSQE)\n";
	}

	// Save the model for later use
	std::cout << "Saving model...\n";
	std::string model_file(data_file + ".xml");
	bool ret = mlpack::data::Save(data_path + model_file, "model", std::move(model), false);
	if (!ret) {
		std::cout << "Serialization ERROR (Save)!\n";
		getchar();
		return -1;
	}

	// Load the model back into a new variable
	std::cout << "Loading model...\n";
	FFN <MeanSquaredError<>, RandomInitialization> modelR;
	ret = mlpack::data::Load(data_path + model_file, "model", modelR);
	if (!ret) {
		std::cout << "Serialization ERROR (Load)!\n";
		getchar();
		return -1;
	}

	// Load test dataset
	mat tempDataset;
	data::Load(data_path + test_data_file, tempDataset, true);
	// Remove decision column
	// We take the full range and then remove the decision column (row in armadillo!)
	mat testX = tempDataset.submat(0, 0, tempDataset.n_rows - 1, tempDataset.n_cols - 1);
	testX.shed_row(iColDecision);

	// Must normalize with the same min, max vector (dn) into the same range as the training data!
	dn.Normalize(testX);
	
	// Getting predictions on test data points .
	mat testPredOut;
	modelR.Predict(testX, testPredOut);

	// Saving the results
	std::ofstream out(data_path + "results.csv");
	out << "Id, Decision" << std::endl;
	for (size_t j = 0; j < testPredOut.n_cols; ++j) {
		if(j>0) out << std::endl;
		out << j + 1 << ", " << testPredOut(j);
	}
	out.close();
	std::cout << "Results were saved to " << data_path + "results.csv\n";

	std::cout << "\n[SAMPLE:END]\n";

	// getchar() prevents the app window closing!
	getchar();
	return 0;
}

