/**
 * @file modelParser.cpp
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
using namespace mlpack::optimization;
using namespace arma;
using namespace std;
using namespace boost::property_tree;

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
   Dataset(){}
   /**
    * Create the Dataset object
    * 
    * Pass the training input dataset and its corresponding output dataset
    * 
    * @param trainX Training input
    * @param trainY Correct output/labels for the training input data
    */
   Dataset(arma::mat& trainX, arma::mat& trainY)
   {
     this->trainX = trainX;
     this->trainY = trainY;
   }
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
           arma::mat& validX, arma::mat& validY)
   {
     this->trainX = trainX;
     this->trainY = trainY;
     this->validX = validX;
     this->validY = validY;
   }
   /**
    * Set the values of the training dataset and its corresponding output
    * 
    * Pass the training input dataset and its corresponding output dataset
    * 
    * @param trainX Training input
    * @param trainY Correct output/labels for the training input data
    */
   void setTrainSet(arma::mat& trainX, arma::mat& trainY)
   {
     this->trainX = trainX;
     this->trainY = trainY;
   }
   /**
    * Set the values of the validation dataset and its corresponding output
    * 
    * Pass the validation input dataset and its corresponding output dataset
    * 
    * @param validX Validation input
    * @param validY Correct output/labels for the validation input data
    */
   void setValidSet(arma::mat& validX, arma::mat& validY)
   {
     this->validX = validX;
     this->validY = validY;
   }
   /**
    * Return the values of the training input dataset
    */
   arma::mat getTrainX()
   {
     return trainX;
   }
   /**
    * Return the values of the training output dataset
    */
   arma::mat getTrainY()
   {
     return trainY;
   }
   /**
    * Return the values of the validation input dataset
    */
   arma::mat getValidX()
   {
     return validX;
   }
   /**
    * Return the values of the validation output dataset
    */
   arma::mat getValidY()
   {
     return validY;
   }
};

/**
 * Print the given stl map where the keys are of type string and
 * values are of type double
 * 
 * @param params The map to be printed
 */
void printMap(map<string, double> params)
{
  map<string, double>::iterator itr;
  for (itr = params.begin(); itr != params.end(); ++itr)
  {
    cout << itr->first << " : " << itr->second << endl;
  }
}

/**
 * Update the values of a given stl map with that of another map 
 * corresponding to the keys that are common
 * 
 * Keys are of type string and values are of type double
 * 
 * @param origParams The map whose values will be updated
 * @param newParams The map whose values will be used to update origParams
 */
void updateParams(map<string, double> &origParams, map<string, double> &newParams)
{
  map<string, double>::iterator itr;
  for (itr = newParams.begin(); itr != newParams.end(); ++itr)
  {
    map<string, double>::iterator itr2 = origParams.find(itr->first);
    itr2->second = newParams.at(itr->first);
  }
  //printMap(origParams);
}

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
arma::Row<size_t> getLabels(const arma::mat& predOut)
{
  arma::Row<size_t> pred(predOut.n_cols);

  // Class of a j-th data point is chosen to be the one with maximum value
  // in j-th column plus 1 (since column's elements are numbered from 0).
  for (size_t j = 0; j < predOut.n_cols; ++j)
  {
    pred(j) = arma::as_scalar(arma::find(
        arma::max(predOut.col(j)) == predOut.col(j), 1)) + 1;
  }

  return pred;
}

/**
 * @author Eugene Freyman
 * Returns the accuracy (percentage of correct answers).
 * @param predLabels predicted labels of data points.
 * @param realY real labels (they are double because we usually read them from
 * CSV file that contain many other double values).
 * @return percentage of correct answers.
 */
double accuracy(arma::Row<size_t> predLabels, const arma::mat& realY)
{
  // Calculating how many predicted classes are coincide with real labels.
  size_t success = 0;
  for (size_t j = 0; j < realY.n_cols; j++) 
    success += predLabels(j) == std::round(realY(j));

  // Calculating percentage of correctly classified data points.
  return (double)success / (double)realY.n_cols * 100.0;
}

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
                int cycles, Dataset& dataset)
{
  arma::mat trainX = dataset.getTrainX();
  arma::mat trainY = dataset.getTrainY();
  arma::mat validX = dataset.getValidX();
  arma::mat validY = dataset.getValidY();
  for(int i = 1; i <= cycles; i++)
  {
    //Uncomment and modify this part for solving a regression problem

    // model.Train(trainX, trainY, optimizer);
    // arma::mat predOut;
    // model.Predict(trainX, predOut);
    // predOut.transform([](double val) { return roundf(val);});
    // double trainAccuracy = 0;
    // for (int j=0; j < trainY.n_cols; j++)
    // {
    //     trainAccuracy += ( (int) predOut[j] == (int) trainY[j]);
    // }
    // trainAccuracy /= (double) trainY.n_cols;
    // model.Predict(validX, predOut);
    // predOut.transform([](double val) { return roundf(val);});
    // double validAccuracy = 0;
    // for (int j=0; j < validY.n_cols; j++)
    // {
    //     validAccuracy += ( (int) predOut[j] == (int) validY[j]);
    // }
    // validAccuracy /= (double) validY.n_cols;
    // cout << "Cycle: " << i << " Training accuracy: " << trainAccuracy <<
    //     " Validation accuracy: " << validAccuracy << endl;

    // Train neural network. If this is the first iteration, weights are
    // random, using current values as starting point otherwise.

    // The following is for a classification problem
    model.Train(trainX, trainY, optimizer);
    mat predOut;
    // Getting predictions on training data points.
    model.Predict(trainX, predOut);
    // Calculating accuracy on training data points.
    Row<size_t> predLabels = getLabels(predOut);
    double trainAccuracy = accuracy(predLabels, trainY);
    // Getting predictions on validating data points.
    model.Predict(validX, predOut);
    // Calculating accuracy on validating data points.
    predLabels = getLabels(predOut);
    double validAccuracy = accuracy(predLabels, validY);

    cout << i << " - accuracy: train = "<< trainAccuracy << "%," <<
      " valid = "<< validAccuracy << "%" <<  endl;
  }
}



/**
 * Create the feedforward network model with the given loss function, 
 * initialization type, optimizer and network architecture
 * 
 * @tparam LossType Type of loss function to use to evaluate the network
 * @tparam InitType Type of initialization to initialize the network parameters
 * @param optimizerType Type of optimizer to use to train the model
 * @param optimizerParams Parameters to use to define the optimizer
 * @param layers The queue of defined layers of type LayerTypes
 * to build the feedforward network
 * @param dataset The Dataset object that contains the training and validation
 * data
 */
template <typename LossType, typename InitType>
void createModel(string optimizerType,
                 map<string, double>& optimizerParams,
                 queue<LayerTypes<> >& layers,
                 Dataset& dataset)
{
  FFN<LossType, InitType> model;
  while (!layers.empty())
  {
    model.Add(layers.front());
    layers.pop();
  }
  map<string, double> origParams;
  origParams["cycles"] = 1;
  string optimizerGroup1[] = {"adadelta", "adagrad", "adam",
      "adamax", "amsgrad", "bigbatchsgd", "momentumsgd",
      "nadam", "nadamax", "nesterovmomentumsgd",
      "optimisticadam", "rmsprop", "sarah", "sgd", "sgdr",
      "snapshotsgdr", "smorms3", "svrg", "spalerasgd"};

  for (string& itr : optimizerGroup1)
  {
    if (itr == optimizerType)
    {
      origParams["stepsize"] = 0.01;
      origParams["batchsize"] = 32;
      origParams["maxiterations"] = 100000;
      origParams["tolerance"] = 1e-5;
      origParams["shuffle"] = true;
      break;
    }
  }

  if (optimizerType == "adadelta")
  {
    origParams["stepsize"] = 1.0;
    origParams["rho"] = 0.95;
    origParams["epsilon"] = 1e-6;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::AdaDelta optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["rho"], origParams["epsilon"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"],
        origParams["resetpolicy"]);
    trainModel<ens::AdaDelta, LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "adagrad")
  {
    origParams["epsilon"] = 1e-8;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::AdaGrad optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["epsilon"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"],
        origParams["resetpolicy"]);
    trainModel<ens::AdaGrad, LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "adam" || optimizerType == "adamax" || 
      optimizerType == "amsgrad" || optimizerType == "optimisticadam" ||
      optimizerType == "nadamax" || optimizerType == "nadam")
  {
    origParams["stepsize"] = 0.001;
    origParams["beta1"] = 0.9;
    origParams["beta2"] = 0.999;
    origParams["epsilon"] = 1e-8; // The docs have it as eps instead of epsilon
    origParams["resetpolicy"] = true;

    updateParams(origParams, optimizerParams);
    if (optimizerType == "adam")
    {
      ens::Adam optimizer(origParams["stepsize"], origParams["batchsize"],
          origParams["beta1"], origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      trainModel<ens::Adam, LossType, InitType>(optimizer, model,
          origParams["cycles"], dataset);
    }
    else if (optimizerType == "adamax")
    {
      ens::AdaMax optimizer(origParams["stepsize"], origParams["batchsize"],
          origParams["beta1"], origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      trainModel<ens::AdaMax, LossType, InitType>(optimizer, model,
          origParams["cycles"], dataset);
    }
    else if (optimizerType == "amsgrad")
    {
      ens::AMSGrad optimizer(origParams["stepsize"], origParams["batchsize"],
          origParams["beta1"], origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      trainModel<ens::AMSGrad, LossType, InitType>(optimizer, model,
          origParams["cycles"], dataset);
    }
    else if (optimizerType == "optimisticadam")
    {
      ens::OptimisticAdam optimizer(origParams["stepsize"],
          origParams["batchsize"], origParams["beta1"],
          origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      trainModel<ens::OptimisticAdam>(optimizer, model, origParams["cycles"],
          dataset);
    }
    else if (optimizerType == "nadamax")
    {
      ens::NadaMax optimizer(origParams["stepsize"], origParams["batchsize"],
          origParams["beta1"], origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      trainModel<ens::NadaMax, LossType, InitType>(optimizer, model,
          origParams["cycles"], dataset);
    }
    else if (optimizerType == "nadam")
    {
      ens::Nadam optimizer(origParams["stepsize"], origParams["batchsize"],
          origParams["beta1"], origParams["beta2"], origParams["epsilon"],
          origParams["maxiterations"], origParams["tolerance"],
          origParams["shuffle"], origParams["resetpolicy"]);
      trainModel<ens::Nadam, LossType, InitType>(optimizer, model,
          origParams["cycles"], dataset);
    }
  }
  else if (optimizerType == "iqn")
  {
    origParams["stepsize"] = 0.01;
    origParams["batchsize"] = 10;
    origParams["maxiterations"] = 100000;
    origParams["tolerance"] = 1e-5;
    updateParams(origParams, optimizerParams);
    ens::IQN optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["maxiterations"], origParams["tolerance"]);
    trainModel<ens::IQN, LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "katyusha")
  {
    origParams["convexity"] = 1.0;
    origParams["lipschitz"] = 10.0;
    origParams["batchsize"] = 10;
    origParams["maxiterations"] = 100000;
    origParams["inneriterations"] = 0;
    origParams["tolerance"] = 1e-5;
    origParams["shuffle"] = true;
    updateParams(origParams, optimizerParams);
    ens::Katyusha optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["maxiterations"], origParams["tolerance"]);
    trainModel<ens::Katyusha, LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "momentumsgd")
  {
    updateParams(origParams, optimizerParams);
    // The MomentumUpdate() parameter can be made modifiable
    ens::MomentumSGD optimizer(origParams["stepsize"],
        origParams["batchsize"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"]);
    trainModel<ens::MomentumSGD, LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "nesterovmomentumsgd")
  {
    updateParams(origParams, optimizerParams);
    // The MomentumUpdate() parameter can be made modifiable
    ens::NesterovMomentumSGD optimizer(origParams["stepsize"],
        origParams["batchsize"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"]);
    trainModel<ens::NesterovMomentumSGD>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "rmsprop")
  {
    origParams["alpha"] = 0.99;
    origParams["epsilon"] = 1e-8;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    // tolerance set to 1e-5
    ens::RMSProp optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["alpha"], origParams["epsilon"],
        origParams["maxiterations"], origParams["tolerance"],
        origParams["shuffle"], origParams["resetpolicy"]);
    trainModel<ens::RMSProp, LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "sarah")
  {
    origParams["inneriterations"] = 0;
    updateParams(origParams, optimizerParams);
    ens::SARAH optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["maxiterations"], origParams["inneriterations"],
        origParams["tolerance"], origParams["shuffle"]);
    trainModel<ens::SARAH, LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "sgd")
  {
    updateParams(origParams, optimizerParams);
    ens::StandardSGD optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["maxiterations"], origParams["tolerance"],
        origParams["shuffle"]);
    trainModel<ens::StandardSGD, LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "sgdr")
  {
    origParams["epochrestart"] = 50;
    origParams["multfactor"] = 2.0;
    origParams["batchsize"] = 1000;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::SGDR<> optimizer(origParams["epochrestart"], origParams["multfactor"],
        origParams["batchsize"], origParams["stepsize"], 
        origParams["maxiterations"], origParams["tolerance"],
        origParams["shuffle"], ens::MomentumUpdate(0.5),
        origParams["resetpolicy"]);
    trainModel<ens::SGDR<> , LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "snapshotsgdr")
  {
    origParams["epochrestart"] = 50;
    origParams["multfactor"] = 2.0;
    origParams["batchsize"] = 1000;
    origParams["snapshots"] = 5;
    origParams["accumulate"] = true;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::SnapshotSGDR<> optimizer(origParams["epochrestart"],
        origParams["multfactor"], origParams["batchsize"],
        origParams["stepsize"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"],
        origParams["snapshots"], origParams["accumulate"],
        ens::MomentumUpdate(0.5), origParams["resetpolicy"]);
    trainModel<ens::SnapshotSGDR<> , LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "smorms3")
  {
    origParams["stepsize"] = 0.001;
    origParams["epsilon"] = 1e-16;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::SMORMS3 optimizer(origParams["stepsize"], origParams["batchsize"], 
        origParams["epsilon"], origParams["maxiterations"],
        origParams["tolerance"], origParams["shuffle"],
        origParams["resetpolicy"]);
    trainModel<ens::SMORMS3, LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "svrg")
  {
    origParams["inneriterations"] = 0;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::SVRG optimizer(origParams["stepsize"], origParams["batchsize"],
        origParams["maxiterations"], origParams["inneriterations"],
        origParams["tolerance"], origParams["shuffle"],
        ens::SVRGUpdate(), ens::NoDecay(), origParams["resetpolicy"]);
    trainModel<ens::SVRG, LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else if (optimizerType == "spalerasgd")
  {
    origParams["lambda"] = 0.01;
    origParams["alpha"] = 0.001;
    origParams["epsilon"] = 1e-6;
    origParams["adaptrate"] = 3.1e-8;
    origParams["resetpolicy"] = true;
    updateParams(origParams, optimizerParams);
    ens::SPALeRASGD<> optimizer(origParams["stepsize"],
        origParams["batchsize"], origParams["maxiterations"],
        origParams["tolerance"], origParams["lambda"],
        origParams["alpha"], origParams["epsilon"],
        origParams["adaptrate"], origParams["shuffle"],
        ens::NoDecay(), origParams["resetpolicy"]);
    trainModel<ens::SPALeRASGD<> , LossType, InitType>(optimizer, model,
        origParams["cycles"], dataset);
  }
  else
  {
    cout << "Invalid optimizer type";
    exit(1);
  }
  
}

/**
 * Determine the loss function to use to evaluate the network given a string
 * that stores the loss function type
 * 
 * @tparam InitType Type of initialization to initialize the network parameters
 * @param lossType Type of loss function to use to evaluate the network
 * @param optimizerType Type of optimizer to use to train the model
 * @param optimizerParams Parameters to use to define the optimizer
 * @param layers The queue of defined layers of type LayerTypes
 * to build the feedforward network
 * @param dataset The Dataset object that contains the training and validation
 * data
 */
template <typename InitType>
void getLossType(string lossType, string optimizerType,
                 map<string, double> optimizerParams,
                 queue<LayerTypes<> >& layers,
                 Dataset& dataset)
{
  if (lossType == "crossentropyerror")
  {
    createModel<CrossEntropyError<>, InitType>(optimizerType,
                                               optimizerParams,
                                               layers,
                                               dataset);
  }
  else if (lossType == "earthmoverdistance")
  {
    //createModel(EarthMoverDistance<>, InitType>();
  }
  else if (lossType == "kldivergence")
  {
    createModel<KLDivergence<>, InitType>(optimizerType,
                                          optimizerParams,
                                          layers,
                                          dataset);
  }
  else if (lossType == "meansquarederror")
  {
    createModel<MeanSquaredError<>, InitType>(optimizerType,
                                              optimizerParams,
                                              layers,
                                              dataset);
  }
  else if (lossType == "negativeloglikelihood")
  {
    createModel<NegativeLogLikelihood<>, InitType>(optimizerType,
                                                   optimizerParams,
                                                   layers,
                                                   dataset);
  }
  else if (lossType == "reconstructionloss")
  {
    //createModel<ReconstructionLoss<>, InitType>(optimizerType,
    //                                            optimizerParams);
  }
  else if (lossType == "sigmoidcrossentropyerror")
  {
    // createModel<SigmoidCrossEntropyError<>, InitType>(optimizerType,
    //                                                   optimizerParams,
    //                                                   layers,
    //                                                   trainX,
    //                                                   trainY);
  }
  else
  {
    cout << "Invalid loss type\n";
    exit(1);
  }
}

/**
 * Determine the initialization type for initializing the network given a
 * string that stores the initialization type
 * 
 * @param initType Type of initialization to initialize the network parameters
 * @param lossType Type of loss function to use to evaluate the network
 * @param initDetails Parameters to use to define the initializer
 * @param optimizerType Type of optimizer to use to train the model
 * @param optimizerParams Parameters to use to define the optimizer
 * @param layers The queue of defined layers of type LayerTypes
 * to build the feedforward network
 * @param dataset The Dataset object that contains the training and validation
 * data
 */
void getInitType(string initType, string lossType,
                 map<string, double> initDetails,
                 string optimizerType, map<string,
                 double> optimizerParams,
                 queue<LayerTypes<> >& layers,
                 Dataset& dataset)
{
  if (initType == "const")
  {
    //initLayer = new ConstInitialization(3.0);
    map<string, double> origParams;
    origParams["initval"];
    //getLossType<ConstInitialization>(lossType);
  }
  else if (initType == "gaussian")
  {
    getLossType<GaussianInitialization>(lossType, optimizerType,
                                        optimizerParams, layers,
                                        dataset);
  }
  else if (initType == "glorot")
  {
    getLossType<GlorotInitialization>(lossType, optimizerType,
                                      optimizerParams, layers,
                                      dataset);
  }
  else if (initType == "he")
  {
    getLossType<HeInitialization>(lossType, optimizerType,
                                  optimizerParams, layers,
                                  dataset);
  }
  else if (initType == "kathirvalavakumar_subavathi")
  {
    //getLossType<KathirvalavakumarSubavathiInitialization>(lossType,
    //                                                      optimizerType,
    //                                                      optimizerParams);
  }
  else if (initType == "lecun_normal")
  {
    getLossType<LecunNormalInitialization>(lossType, optimizerType,
                                           optimizerParams, layers,
                                           dataset);
  }
  else if (initType == "nguyen_widrow")
  {
    getLossType<NguyenWidrowInitialization>(lossType, optimizerType,
                                            optimizerParams, layers,
                                            dataset);
  }
  else if (initType == "oivs")
  {
    //getLossType<OivsInitialization>(lossType, optimizerType, optimizerParams);
  }
  else if (initType == "orthogonal")
  {
    getLossType<OrthogonalInitialization>(lossType, optimizerType,
                                          optimizerParams, layers,
                                          dataset);
  }
  else if (initType == "random")
  {
    getLossType<RandomInitialization>(lossType, optimizerType,
                                      optimizerParams, layers,
                                      dataset);
  }
  else
  {
    cout << "Invalid initialization type";
    exit(1);
  }
}

void testMaps()
{
  map<string, double> param1;
  map<string, double> param2;
  param1["id1"] = 1.0;
  param1["id2"] = 2.0;
  param1["id3"] = 3.0;
  param1["id4"] = 4.0;

  param2["id1"] = 11.0;
  param2["id2"] = 12.0;
  param2["id6"] = 13.0;
  param2["id7"] = 14.0;

  updateParams(param1, param2);
  printMap(param1);
}

/**
 * Determine the layer type to be added to a feedforward network given a
 * string containing the type and a map containing the parameters
 * 
 * @param layerType Type of layer that is to be defined
 * @param layerParams Map containing the parameters of the layer to be defined
 * @return A LayerTypes<> object that is of the given type and is 
 * initialized by the given parameters
 */
LayerTypes<> getNetworkReference(string layerType, map<string, double>& layerParams)
{
  map<string, double> origParams;
  LayerTypes<> layer;

  if (layerType == "alphadropout")
  {
    origParams["ratio"] = 0.5;
    // alphadash is the default value of -alpha*lambda
    origParams["alphadash"] = -1.758099340847376624;
    updateParams(origParams, layerParams);
    layer = new AlphaDropout<>(origParams["ratio"], origParams["alphadash"]);
  }
  else if (layerType == "batchnorm")
  {
    layer = new BatchNorm<>();
  }
  else if (layerType == "constant")
  {
    origParams["outsize"];
    origParams["scalar"] = 0.0;
    layer = new Constant<>(origParams["outsize"], origParams["scalar"]);
  }
  else if (layerType == "convolution")
  {
    origParams["insize"];
    origParams["outsize"];
    origParams["kw"];
    origParams["kh"];
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["padw"] = 0;
    origParams["padh"] = 0;
    origParams["inputwidth"] = 0;
    origParams["inputheight"] = 0;
    updateParams(origParams, layerParams);
    layer = new Convolution<>(origParams["insize"], origParams["outsize"],
        origParams["kw"], origParams["kh"], origParams["dw"], 
        origParams["dh"], origParams["padw"], origParams["padh"],
        origParams["inputwidth"], origParams["inputheight"]);
  }
  else if (layerType == "dropconnect")
  {
    // origParams["insize"];
    // origParams["outsize"];
    // origParams["ratio"] = 0.5;
    // updateParams(origParams, layerParams);
    // layer = new DropConnect<>();
  }
  else if (layerType == "dropout")
  {
    origParams["ratio"] = 0.5;
    updateParams(origParams, layerParams);
    layer = new Dropout<>(origParams["ratio"]);
  }
  else if (layerType == "fastlstm")
  {
    //origParams = {{""}}
  }
  else if (layerType == "gru")
  {
  }
  else if (layerType == "layernorm")
  {
    layer = new LayerNorm<>();
  }
  else if (layerType == "linearnobias")
  {
    origParams["insize"];
    origParams["outsize"];
    updateParams(origParams, layerParams);
    layer = new LinearNoBias<>(origParams["insize"], origParams["outsize"]);
  }
  else if (layerType == "linear")
  {
    origParams["insize"];
    origParams["outsize"];
    updateParams(origParams, layerParams);
    layer = new Linear<>(origParams["insize"], origParams["outsize"]);
  }
  else if (layerType == "maxpooling")
  {
    origParams["kw"];
    origParams["kh"];
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["floor"] = 1; // 1 for true, 0 for false
    updateParams(origParams, layerParams);
    layer = new MaxPooling<>(origParams["kw"], origParams["kh"],
        origParams["dw"], origParams["dh"], origParams["floor"]);
  }
  else if (layerType == "meanpooling")
  {
    origParams["kw"];
    origParams["kh"];
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["floor"] = true;
    updateParams(origParams, layerParams);
    layer = new MeanPooling<>(origParams["kw"], origParams["kh"],
        origParams["dw"], origParams["dh"], origParams["floor"]);
  }
  else if (layerType == "multiplyconstant")
  {
    origParams["scalar"] = 1.0;
    updateParams(origParams, layerParams);
    layer = new MultiplyConstant<>(origParams["scalar"]);
  }
  else if (layerType == "recurrent")
  {

  }
  else if (layerType == "transposedconvolution")
  {
    origParams["insize"];
    origParams["outsize"];
    origParams["kw"];
    origParams["kh"];
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["padw"] = 0;
    origParams["padh"] = 0;
    origParams["inputwidth"] = 0;
    origParams["inputheight"] = 0;
    updateParams(origParams, layerParams);
    layer = new TransposedConvolution<>(origParams["insize"],
        origParams["outsize"], origParams["kw"],origParams["kh"],
        origParams["dw"], origParams["dh"], origParams["padw"],
        origParams["padh"], origParams["inputwidth"],
        origParams["inputheight"]);
  }
  else if (layerType == "identity")
  {
    layer = new IdentityLayer<>();
  }
  else if (layerType == "logistic")
  {
    //layer = LogisticFunction();
  }
  else if (layerType == "rectifier" || layerType == "relu")
  {
    layer = new ReLULayer<>();
  }
  else if (layerType == "softplus")
  {

  }
  else if (layerType == "softsign")
  {
    //layer = SoftsignFunction;
  }
  else if (layerType == "swish")
  {
    //layer = new Swish
  }
  else if (layerType == "tanh")
  {
    layer = new TanHLayer<>();
  }
  else if (layerType == "elu")
  {
    origParams["alpha"];
    updateParams(origParams, layerParams);
    layer = new ELU<>(origParams["alpha"]);
  }
  else if (layerType == "selu")
  {
    layer = new SELU();
  }
  else if (layerType == "hardtanh")
  {
    origParams["maxvalue"] = 1;
    origParams["minvalue"] = -1;
    updateParams(origParams, layerParams);
    layer = new HardTanH<>(origParams["maxvalue"], origParams["minvalue"]);
  }
  else if (layerType == "leakyrelu")
  {
    origParams["alpha"] = 0.03;
    updateParams(origParams, layerParams);
    layer = new LeakyReLU<>(origParams["alpha"]);
  }
  else if (layerType == "prelu")
  {
    origParams["alpha"] = 0.03; // userAlpha
    layer = new PReLU<>(origParams["alpha"]);
  }
  else if (layerType == "sigmoid")
  {
    layer = new SigmoidLayer<>();
  }
  else if (layerType == "softmax" || layerType == "logsoftmax")
  {
    layer = new LogSoftMax<>();
  }
  else
  {
    cout << "Invalid layer type : " << layerType;
    exit(1);
  }
  return layer;
}

/**
 * Traverse the given property tree and determine the loss function, 
 * initializer, optimizer and network architecture as given by the user
 * 
 * @param tree Property tree to use to extract the network features
 * @param dataset The Dataset object that contains the training and validation
 * data
 * @param inSize The input size of the first layer
 */ 
void traverseModel(const ptree& tree, Dataset& dataset, double& inSize)
{
  const ptree &loss = tree.get_child("loss");
  const ptree &init = tree.get_child("init");
  const ptree &optimizer = tree.get_child("optimizer");
  const ptree &network = tree.get_child("network");
  queue<LayerTypes<> > layers;

  string lossType = loss.get_value<string>();
  cout << "Loss type : " << lossType << "\n\n";

  map<string, double> initDetails;
  string initType;
  BOOST_FOREACH (ptree::value_type const &v, init.get_child(""))
  {
    const ptree &attributes = v.second;
    if (v.first == "type")
    {
      initType = attributes.get_value<string>();
    }
    else
    {
      initDetails[v.first] = attributes.get_value<double>();
    }
  }
  cout << "Init details:\ntype : " << initType << endl;
  printMap(initDetails);
  
  map<string, double> optimizerDetails;
  string optimizerType;
  BOOST_FOREACH (ptree::value_type const &v, optimizer.get_child(""))
  {
    const ptree &attributes = v.second;
    if (v.first == "type")
    {
      optimizerType = attributes.get_value<string>();
    }
    else
    {
      optimizerDetails[v.first] = attributes.get_value<double>();
    }
  }
  cout << "\nOptimizer details:\ntype : " << optimizerType << endl;
  printMap(optimizerDetails);
  cout << "\nNetwork details:\n\n";
  BOOST_FOREACH (ptree::value_type const &v, network.get_child(""))
  {
    const ptree &layerWhole = v.second;
    map<string, double> params;
    string layerType;
    BOOST_FOREACH (ptree::value_type const &v2, layerWhole.get_child(""))
    {
      const ptree &layerInner = v2.second;
      //cout << v2.first << "\t";
      //cout << layerInner.get_value<string>() << endl;
      string key = boost::erase_all_copy(v2.first, "_");
      boost::to_lower(key);
      if (key == "type")
      {
        layerType = layerInner.get_value<string>();
      }
      else if (key == "units")
      {
        params["insize"] = inSize;
        inSize = params["outsize"] = layerInner.get_value<double>();
      }
      else
      {
        if(key == "outsize")
          inSize = layerInner.get_value<double>();
        params[key] = layerInner.get_value<double>();
      }
    }
    cout << "type : " << layerType << endl;
    printMap(params);
    layers.push(getNetworkReference(layerType, params));
    cout << endl;
  }
  getInitType(initType, lossType, initDetails, optimizerType,
      optimizerDetails, layers, dataset);
}

/**
 * Create a propert tree from the given json file
 * 
 * @param fileName Path to the json file from which the network
 * properties would be loaded
 * @param dataset The Dataset object that contains the training and validation
 * data
 * @param inSize The input size of the first layer
 */
boost::property_tree::ptree loadProperties(string fileName, Dataset& dataset,
                                           double inSize)
{
  ptree pt;
  read_json(fileName, pt);
  traverseModel(pt, dataset, inSize);
  return pt;
}

/** The final implementation of this file would not have a main method. This is 
 * merely to ease testing. The following include satement can hence be 
 * removed later
*/
#include <mlpack/core/data/split_data.hpp>
int main()
{
  //arma::mat trainX, trainY;
  string fileName = "network3.json";
  arma::mat dataset2;
  data::Load("train.csv", dataset2, true);
  cout << "Data loaded" << "\n\n";
  dataset2 = dataset2.submat(0, 1, dataset2.n_rows - 1, dataset2.n_cols - 1);
  arma::mat train, valid;
  data::Split(dataset2, train, valid, 0.1);
  arma::mat trainX = normalise(train.submat(1, 0, train.n_rows-1, train.n_cols-1));
  arma::mat trainY = train.row(0) + 1;
  arma::mat validX = normalise(valid.submat(1, 0, valid.n_rows-1, valid.n_cols-1));
  arma::mat validY = valid.row(0) + 1;
  Dataset dataset(trainX, trainY, validX, validY);
  loadProperties(fileName, dataset, trainX.n_rows);
  //testMaps();
  return 0;
}
