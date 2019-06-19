/**
 * @file model_parser.hpp
 * @author Sreenik Seal
 *
 * Implementation of a parser to parse json files containing 
 * user-defined model details to train neural networks
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <queue>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp>
#include <mlpack/methods/ann/init_rules/lecun_normal_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/init_rules/oivs_init.hpp>
#include <mlpack/methods/ann/init_rules/orthogonal_init.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
//#include <mlpack/methods/ann/loss_functions/earth_mover_distance.hpp>
#include <mlpack/methods/ann/loss_functions/kl_divergence.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp>
//#include <mlpack/methods/ann/loss_functions/reconstruction_loss.hpp>
#include <mlpack/methods/ann/activation_functions/softsign_function.hpp>
#include <mlpack/methods/ann/activation_functions/swish_function.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <ensmallen.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>

using namespace mlpack;
using namespace mlpack::ann;
//using namespace mlpack::optimization;
using namespace arma;
using namespace std;
using namespace boost::property_tree;

bool error = false;
/**
 * Implementation of the entire dataset consisting of the training data
 * and validation data
 */
class Dataset
{
 private:
   arma::mat trainX, trainY, validX, validY;
 public:
   /**
    * Create the Dataset object
    */
   Dataset();
   /**
    * Create the Dataset object
    * 
    * Pass the training input dataset and its corresponding output dataset
    * 
    * @param trainX Training input
    * @param trainY Correct output/labels for the training input data
    */
   Dataset(arma::mat& trainX, arma::mat& trainY);
   /**
    * Create the Dataset object
    * 
    * Pass the training input dataset, validation input dataset and their
    * respective output datasets
    * 
    * @param trainX Training input
    * @param trainY Correct ouput/labels for the training input data
    * @param validX Validation input
    * @param validY Correct output/labels for the validation input data
    */
   Dataset(arma::mat& trainX, arma::mat& trainY,
           arma::mat& validX, arma::mat& validY);
   /**
    * Set the values of the training dataset and its corresponding output
    * 
    * Pass the training input dataset and its corresponding output dataset
    * 
    * @param trainX Training input
    * @param trainY Correct output/labels for the training input data
    */
   void setTrainSet(arma::mat& trainX, arma::mat& trainY);
   /**
    * Set the values of the validation dataset and its corresponding output
    * 
    * Pass the validation input dataset and its corresponding output dataset
    * 
    * @param validX Validation input
    * @param validY Correct output/labels for the validation input data
    */
   void setValidSet(arma::mat& validX, arma::mat& validY);
   /**
    * Return the values of the training input dataset
    */
   arma::mat getTrainX();
   /**
    * Return the values of the training output dataset
    */
   arma::mat getTrainY();
   /**
    * Return the values of the validation input dataset
    */
   arma::mat getValidX();
   /**
    * Return the values of the validation output dataset
    */
   arma::mat getValidY();
};

/**
 * Print the given stl map where the keys are of type string and
 * values are of type double
 * 
 * @param params The map to be printed
 */
void printMap(map<string, double> params);

/**
 * Update the values of a given stl map with that of another map 
 * corresponding to the keys that are common
 * 
 * Keys are of type string and values are of type double
 * 
 * @param origParams The map whose values will be updated
 * @param newParams The map whose values will be used to update origParams
 */
void updateParams(map<string, double> &origParams, map<string, double> &newParams);

/**
 * @author Eugene Freyman
 * Returns labels bases on predicted probability (or log of probability)
 * of classes.
 * @param predOut matrix contains probabilities (or log of probability) of
 * classes. Each row corresponds to a certain class, each column corresponds
 * to a data point.
 * @return a row vector of data point's classes. The classes starts from 1 to
 * the number of rows in input matrix.
 */
arma::Row<size_t> getLabels(const arma::mat& predOut);

/**
 * @author Eugene Freyman
 * Returns the accuracy (percentage of correct answers).
 * @param predLabels predicted labels of data points.
 * @param realY real labels (they are double because we usually read them from
 * CSV file that contain many other double values).
 * @return percentage of correct answers.
 */
double accuracy(arma::Row<size_t> predLabels, const arma::mat& realY);

/**
 * Train the feedforward network with the given training data and test it
 * against the given validation data for a given number of cycles
 * 
 * @tparam OptimizerType Type of optimizer to use to train the model
 * @tparam LossType Type of loss function to use to evaluate the network
 * @tparam InitType Type of initialization to initialize the network parameter
 * @param optimizer Optimizer to use to train the model
 * @param model FFN object with the given loss type and initialization type
 * @param cycles The number of cycles the network will be trained
 * @param dataset The Dataset object that contains the training and validation
 * data
 */
template <typename OptimizerType, typename LossType, typename InitType>
void trainModel(OptimizerType optimizer, FFN<LossType, InitType> model,
                int cycles, Dataset& dataset);



/**
 * Create the feedforward network model with the given loss function, 
 * initialization type, optimizer and network architecture
 * 
 * @tparam LossType Type of loss function to use to evaluate the network
 * @tparam InitType Type of initialization to initialize the network parameters
 * @param loss Loss function used to evaluate the network
 * @param init Initializer to initialize the network
 * @param optimizerType Type of optimizer to use to train the model
 * @param optimizerParams Parameters to use to define the optimizer
 * @param layers The queue of defined layers of type LayerTypes
 * to build the feedforward network
 * @param dataset The Dataset object that contains the training and validation
 * data
 */
template <typename LossType, typename InitType>
void createModel(LossType& loss,
                 InitType& init,
                 string& optimizerType,
                 map<string, double>& optimizerParams,
                 queue<LayerTypes<> >& layers,
                 Dataset& dataset);

/**
 * Determine the loss function to use to evaluate the network given a string
 * that stores the loss function type
 * 
 * @tparam InitType Type of initialization to initialize the network parameters
 * @param init Initializer to initialize the network
 * @param lossType Type of loss function to use to evaluate the network
 * @param optimizerType Type of optimizer to use to train the model
 * @param lossParams Parameters used to define the loss function
 * @param optimizerParams Parameters to use to define the optimizer
 * @param layers The queue of defined layers of type LayerTypes
 * to build the feedforward network
 * @param dataset The Dataset object that contains the training and validation
 * data
 */
template <typename InitType>
void getLossType(InitType& init,
                 string& lossType, string& optimizerType,
                 map<string, double>& lossParams,
                 map<string, double>& optimizerParams,
                 queue<LayerTypes<> >& layers,
                 Dataset& dataset);

/**
 * Determine the initialization type for initializing the network given a
 * string that stores the initialization type
 * 
 * @param initType Type of initialization to initialize the network parameters
 * @param lossType Type of loss function to use to evaluate the network
 * @param initParams Parameters to use to define the initializer
 * @param lossParams Parameters to use to define the loss function
 * @param optimizerType Type of optimizer to use to train the model
 * @param optimizerParams Parameters to use to define the optimizer
 * @param layers The queue of defined layers of type LayerTypes
 * to build the feedforward network
 * @param dataset The Dataset object that contains the training and validation
 * data
 */
void getInitType(string& initType, string& lossType,
                 map<string, double>& initParams,
                 map<string, double>& lossParams,
                 string& optimizerType, map<string,
                 double>& optimizerParams,
                 queue<LayerTypes<> >& layers,
                 Dataset& dataset);

/**
 * Determine the layer type to be added to a feedforward network given a
 * string containing the type and a map containing the parameters
 * 
 * @param layerType Type of layer that is to be defined
 * @param layerParams Map containing the parameters of the layer to be defined
 * @return A LayerTypes<> object that is of the given type and is 
 * initialized by the given parameters
 */
LayerTypes<> getNetworkReference(string& layerType, map<string, double>& layerParams);

/**
 * Traverse the given property tree and determine the loss function, 
 * initializer, optimizer and network architecture as given by the user
 * 
 * @param tree Property tree to use to extract the network features
 * @param dataset The Dataset object that contains the training and validation
 * data
 * @param inSize The input size of the first layer
 */ 
void traverseModel(const ptree& tree, Dataset& dataset, double& inSize);

/**
 * Create a property tree from the given json file
 * 
 * @param fileName Path to the json file from which the network
 * properties would be loaded
 * @param dataset The Dataset object that contains the training and validation
 * data
 * @param inSize The input size of the first layer
 */
boost::property_tree::ptree loadProperties(string& fileName, Dataset& dataset,
                                           double inSize);

/** The final implementation of this file would not have a main method. This is 
 * merely to ease testing. The following include statement can hence be 
 * removed later
*/


#include <mlpack/core/data/split_data.hpp>
int testParser();
