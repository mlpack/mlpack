/**
 * @file model_parser_impl.hpp
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

#ifndef MLPACK_METHODS_ANN_MODEL_PARSER_IMPL_HPP
#define MLPACK_METHODS_ANN_MODEL_PARSER_IMPL_HPP

#include "model_parser.hpp"

using namespace mlpack;
using namespace ann;

Dataset::Dataset(){}

Dataset::Dataset(arma::mat& trainX, arma::mat& trainY)
{
  this->trainX = trainX;
  this->trainY = trainY;
}

Dataset::Dataset(arma::mat& trainX, arma::mat& trainY,
        arma::mat& validX, arma::mat& validY)
{
  this->trainX = trainX;
  this->trainY = trainY;
  this->validX = validX;
  this->validY = validY;
}

void Dataset::setTrainSet(arma::mat& trainX, arma::mat& trainY)
{
  this->trainX = trainX;
  this->trainY = trainY;
}

void Dataset::setValidSet(arma::mat& validX, arma::mat& validY)
{
  this->validX = validX;
  this->validY = validY;
}

arma::mat Dataset::getTrainX()
{
  return trainX;
}

arma::mat Dataset::getTrainY()
{
  return trainY;
}

arma::mat Dataset::getValidX()
{
  return validX;
}

arma::mat Dataset::getValidY()
{
  return validY;
}

void printMap(std::map<std::string, double> params)
{
  std::map<std::string, double>::iterator itr;
  for (itr = params.begin(); itr != params.end(); ++itr)
  {
    Log::Info << itr->first << " : " << itr->second << "\n";
  }
}

void updateParams(std::map<std::string, double> &origParams,
                  std::map<std::string, double> &newParams)
{
  std::map<std::string, double>::iterator itr;
  bool error = false;
  for (itr = origParams.begin(); itr != origParams.end(); ++itr)
  {
    std::map<std::string, double>::iterator itr2 = newParams.find(itr->first);
    if (itr2 == newParams.end() && isnan(itr->second))
    {
      Log::Info << "Required parameter: " << itr->first << "\n";
      error = true;
    }
    else if (itr2 != newParams.end())
      itr->second = newParams.at(itr->first);
  }
  if (error)
    Log::Fatal << "Required parameters missing" << "\n";
}


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

double accuracy(arma::Row<size_t> predLabels, const arma::mat& realY)
{
  // Calculating how many predicted classes are coincide with real labels.
  size_t success = 0;
  for (size_t j = 0; j < realY.n_cols; j++)
    success += predLabels(j) == std::round(realY(j));

  // Calculating percentage of correctly classified data points.
  return (double)success / (double)realY.n_cols * 100.0;
}

template <typename OptimizerType, typename LossType, typename InitType>
void trainModel(OptimizerType optimizer, FFN<LossType, InitType> model,
                int cycles, Dataset& dataset)
{
  arma::mat trainX = dataset.getTrainX();
  arma::mat trainY = dataset.getTrainY();
  arma::mat validX = dataset.getValidX();
  arma::mat validY = dataset.getValidY();
  for (int i = 1; i <= cycles; i++)
  {
    // Uncomment and modify this part for solving a regression problem

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
    // Log::Info << "Cycle: " << i << " Training accuracy: " << trainAccuracy <<
    //     " Validation accuracy: " << validAccuracy << "\n";

    // Train neural network. If this is the first iteration, weights are
    // random, using current values as starting point otherwise.

    // The following is for a classification problem
    model.Train(trainX, trainY, optimizer);
    arma::mat predOut;
    // Getting predictions on training data points.
    model.Predict(trainX, predOut);
    // Calculating accuracy on training data points.
    arma::Row<size_t> predLabels = getLabels(predOut);
    double trainAccuracy = accuracy(predLabels, trainY);
    // Getting predictions on validating data points.
    model.Predict(validX, predOut);
    // Calculating accuracy on validating data points.
    predLabels = getLabels(predOut);
    double validAccuracy = accuracy(predLabels, validY);

    Log::Info << i << " - accuracy: train = "<< trainAccuracy << "%," <<
      " valid = "<< validAccuracy << "%" <<  "\n";
  }
}

template <typename LossType, typename InitType>
void createModel(LossType& loss,
                 InitType& init,
                 std::string& optimizerType,
                 std::map<std::string, double>& optimizerParams,
                 std::queue<LayerTypes<> >& layers,
                 Dataset& dataset)
{
  FFN<LossType, InitType> model(loss, init);
  while (!layers.empty())
  {
    model.Add(layers.front());
    layers.pop();
  }
  std::map<std::string, double> origParams;
  origParams["cycles"] = 1;
  std::string optimizerGroup1[] = {"adadelta", "adagrad", "adam",
      "adamax", "amsgrad", "bigbatchsgd", "momentumsgd",
      "nadam", "nadamax", "nesterovmomentumsgd",
      "optimisticadam", "rmsprop", "sarah", "sgd", "sgdr",
      "snapshotsgdr", "smorms3", "svrg", "spalerasgd"};

  for (std::string& itr : optimizerGroup1)
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
    Log::Fatal << "Invalid optimizer type";
  }
}

template <typename InitType>
void getLossType(InitType& init,
                 std::string& lossType, std::string& optimizerType,
                 std::map<std::string, double>& lossParams,
                 std::map<std::string, double>& optimizerParams,
                 std::queue<LayerTypes<> >& layers,
                 Dataset& dataset)
{
  std::map<std::string, double> origParams;
  if (lossType == "crossentropyerror")
  {
    origParams["eps"] = 1e-10;
    updateParams(origParams, lossParams);
    CrossEntropyError<> loss(origParams["eps"]);
    createModel<CrossEntropyError<>, InitType>(loss,
                                               init,
                                               optimizerType,
                                               optimizerParams,
                                               layers,
                                               dataset);
  }
  else if (lossType == "earthmoverdistance")
  {
    EarthMoverDistance<> loss;
    createModel<EarthMoverDistance<>, InitType>(loss,
                                              init,
                                              optimizerType,
                                              optimizerParams,
                                              layers,
                                              dataset);
  }
  else if (lossType == "kldivergence")
  {
    origParams["takemean"] = false;
    updateParams(origParams, lossParams);
    KLDivergence<> loss(origParams["takemean"]);
    createModel<KLDivergence<>, InitType>(loss,
                                          init,
                                          optimizerType,
                                          optimizerParams,
                                          layers,
                                          dataset);
  }
  else if (lossType == "meansquarederror")
  {
    MeanSquaredError<> loss;
    createModel<MeanSquaredError<>, InitType>(loss,
                                              init,
                                              optimizerType,
                                              optimizerParams,
                                              layers,
                                              dataset);
  }
  else if (lossType == "negativeloglikelihood")
  {
    NegativeLogLikelihood<> loss;
    createModel<NegativeLogLikelihood<>, InitType>(loss,
                                                   init,
                                                   optimizerType,
                                                   optimizerParams,
                                                   layers,
                                                   dataset);
  }
  else if (lossType == "reconstructionloss")
  {
    ReconstructionLoss<> loss;
    createModel<ReconstructionLoss<>, InitType>(loss,
                                                init,
                                                optimizerType,
                                                optimizerParams,
                                                layers,
                                                dataset);
  }
  else if (lossType == "sigmoidcrossentropyerror")
  {
    SigmoidCrossEntropyError<> loss;
    createModel<SigmoidCrossEntropyError<>, InitType>(loss,
                                                      init,
                                                      optimizerType,
                                                      optimizerParams,
                                                      layers,
                                                      dataset);
  }
  else
  {
    Log::Fatal << "Invalid loss type\n";
  }
}

void getInitType(std::string& initType, std::string& lossType,
                 std::map<std::string, double>& initParams,
                 std::map<std::string, double>& lossParams,
                 std::string& optimizerType, std::map<std::string,
                 double>& optimizerParams,
                 std::queue<LayerTypes<> >& layers,
                 Dataset& dataset)
{
  std::map<std::string, double> origParams;
  if (initType == "const")
  {
    origParams["initval"] = NAN;
    updateParams(origParams, initParams);
    ConstInitialization init(origParams["initval"]);
    getLossType<ConstInitialization>(init, lossType,
                                     optimizerType, lossParams,
                                     optimizerParams,
                                     layers, dataset);
  }
  else if (initType == "gaussian")
  {
    origParams["mean"] = 0;
    origParams["variance"] = 1;
    GaussianInitialization init(origParams["mean"], origParams["variance"]);
    getLossType<GaussianInitialization>(init, lossType, optimizerType,
                                        lossParams, optimizerParams, layers,
                                        dataset);
  }
  else if (initType == "glorot")
  {
    GlorotInitialization init;
    getLossType<GlorotInitialization>(init, lossType, optimizerType,
                                      lossParams, optimizerParams, layers,
                                      dataset);
  }
  else if (initType == "he")
  {
    HeInitialization init;
    getLossType<HeInitialization>(init, lossType, optimizerType,
                                  lossParams, optimizerParams, layers,
                                  dataset);
  }
  // else if (initType == "kathirvalavakumar_subavathi")
  // {
  //   getLossType<KathirvalavakumarSubavathiInitialization>(lossType,
  //                                                        optimizerType,
  //                                                        optimizerParams);
  // }
  else if (initType == "lecun_normal")
  {
    LecunNormalInitialization init;
    getLossType<LecunNormalInitialization>(init, lossType, optimizerType,
                                           lossParams, optimizerParams, layers,
                                           dataset);
  }
  else if (initType == "nguyen_widrow")
  {
    origParams["lowerbound"] = -0.5;
    origParams["upperbound"] = 0.5;
    updateParams(origParams, initParams);
    NguyenWidrowInitialization init(origParams["lowerbound"],
                                    origParams["upperbound"]);
    getLossType<NguyenWidrowInitialization>(init, lossType, optimizerType,
                                            lossParams, optimizerParams,
                                            layers, dataset);
  }
  else if (initType == "oivs")
  {
    origParams["epsilon"] = 0.1;
    origParams["k"] = 5;
    origParams["gamma"] = 0.9;
    updateParams(origParams, initParams);
    OivsInitialization<> init(origParams["epsilon"], origParams["k"],
                              origParams["gamma"]);
    getLossType<OivsInitialization<>>(init, lossType, optimizerType,
                                      lossParams, optimizerParams,
                                      layers, dataset);
  }
  else if (initType == "orthogonal")
  {
    origParams["gain"] = 1.0;
    updateParams(origParams, initParams);
    OrthogonalInitialization init(origParams["gain"]);
    getLossType<OrthogonalInitialization>(init, lossType, optimizerType,
                                          lossParams, optimizerParams, layers,
                                          dataset);
  }
  else if (initType == "random")
  {
    origParams["lowerbound"] = -1.0;
    origParams["upperbound"] = 1.0;
    updateParams(origParams, initParams);
    RandomInitialization init(origParams["lowerbound"],
                                    origParams["upperbound"]);
    getLossType<RandomInitialization>(init, lossType, optimizerType,
                                      lossParams, optimizerParams, layers,
                                      dataset);
  }
  else
  {
    Log::Fatal << "Invalid initialization type";
  }
}

LayerTypes<> getNetworkReference(std::string& layerType,
                                 std::map<std::string, double>& layerParams)
{
  std::map<std::string, double> origParams;
  LayerTypes<> layer;

  if (layerType == "atrousconvolution")
  {
    origParams["insize"] = NAN;
    origParams["outsize"] = NAN;
    origParams["kw"] = NAN;
    origParams["kh"] = NAN;
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["padw"] = 0;
    origParams["padh"] = 0;
    origParams["inputwidth"] = 0;
    origParams["inputheight"] = 0;
    origParams["dilationw"] = 1;
    origParams["dilationh"] = 1;
    updateParams(origParams, layerParams);
    layer = new AtrousConvolution<>(origParams["insize"],
        origParams["outsize"], origParams["kw"], origParams["kh"],
        origParams["dw"], origParams["dh"], origParams["padw"],
        origParams["padh"], origParams["inputwidth"],
        origParams["inputheight"], origParams["dilationw"],
        origParams["dilationh"]);
  }
  else if (layerType == "alphadropout")
  {
    origParams["ratio"] = 0.5;
    // alphadash is the default value of -alpha*lambda
    origParams["alphadash"] = -1.758099340847376624;
    updateParams(origParams, layerParams);
    layer = new AlphaDropout<>(origParams["ratio"], origParams["alphadash"]);
  }
  else if (layerType == "batchnorm")
  {
    origParams["size"] = NAN;
    origParams["eps"] = 1e-8;;
    updateParams(origParams, layerParams);
    layer = new BatchNorm<>(origParams["size"], origParams["eps"]);
  }
  else if (layerType == "constant")
  {
    origParams["outsize"] = NAN;
    origParams["scalar"] = 0.0;
    layer = new Constant<>(origParams["outsize"], origParams["scalar"]);
  }
  else if (layerType == "convolution")
  {
    origParams["insize"] = NAN;
    origParams["outsize"] = NAN;
    origParams["kw"] = NAN;
    origParams["kh"] = NAN;
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
    origParams["insize"];
    origParams["outsize"];
    origParams["ratio"] = 0.5;
    updateParams(origParams, layerParams);
    layer = new DropConnect<>(origParams["insize"], origParams["outsize"],
                              origParams["ratio"]);
  }
  else if (layerType == "dropout")
  {
    origParams["ratio"] = 0.5;
    updateParams(origParams, layerParams);
    layer = new Dropout<>(origParams["ratio"]);
  }
  else if (layerType == "layernorm")
  {
    layer = new LayerNorm<>();
  }
  else if (layerType == "linearnobias")
  {
    origParams["insize"] = NAN;
    origParams["outsize"] = NAN;
    updateParams(origParams, layerParams);
    layer = new LinearNoBias<>(origParams["insize"], origParams["outsize"]);
  }
  else if (layerType == "linear")
  {
    origParams["insize"] = NAN;
    origParams["outsize"] = NAN;
    updateParams(origParams, layerParams);
    layer = new Linear<>(origParams["insize"], origParams["outsize"]);
  }
  else if (layerType == "maxpooling")
  {
    origParams["kw"] = NAN;
    origParams["kh"] = NAN;
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["floor"] = 1; // 1 for true, 0 for false
    updateParams(origParams, layerParams);
    layer = new MaxPooling<>(origParams["kw"], origParams["kh"],
        origParams["dw"], origParams["dh"], origParams["floor"]);
  }
  else if (layerType == "meanpooling")
  {
    origParams["kw"] = NAN;
    origParams["kh"] = NAN;
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
  else if (layerType == "transposedconvolution")
  {
    origParams["insize"] = NAN;
    origParams["outsize"] = NAN;
    origParams["kw"] = NAN;
    origParams["kh"] = NAN;
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["padw"] = 0;
    origParams["padh"] = 0;
    origParams["inputwidth"] = 0;
    origParams["inputheight"] = 0;
    updateParams(origParams, layerParams);
    layer = new TransposedConvolution<>(origParams["insize"],
        origParams["outsize"], origParams["kw"], origParams["kh"],
        origParams["dw"], origParams["dh"], origParams["padw"],
        origParams["padh"], origParams["inputwidth"],
        origParams["inputheight"]);
  }
  else if (layerType == "identity")
  {
    layer = new IdentityLayer<>();
  }
  else if (layerType == "rectifier" || layerType == "relu")
  {
    layer = new ReLULayer<>();
  }
  // else if (layerType == "softplus")
  // {
  // }
  // else if (layerType == "softsign")
  // {
  //   // layer = SoftsignFunction;
  // }
  // else if (layerType == "swish")
  // {
  //   // layer = new Swish
  // }
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
    Log::Fatal << "Invalid layer type : " << layerType;
  }
  return layer;
}

void traverseModel(const boost::property_tree::ptree& tree,
                   Dataset& dataset, double& inSize)
{
  const boost::property_tree::ptree &loss = tree.get_child("loss");
  const boost::property_tree::ptree &init = tree.get_child("init");
  const boost::property_tree::ptree &optimizer = tree.get_child("optimizer");
  const boost::property_tree::ptree &network = tree.get_child("network");
  std::queue<LayerTypes<> > layers;

  std::map<std::string, double> lossParams;
  std::string lossType;
  BOOST_FOREACH(boost::property_tree::ptree::value_type const &v,
                loss.get_child(""))
  {
    const boost::property_tree::ptree &attributes = v.second;
    if (v.first == "type")
    {
      lossType = attributes.get_value<std::string>();
    }
    else
    {
      lossParams[v.first] = attributes.get_value<double>();
    }
  }
  Log::Info << "Loss details:\ntype : " << lossType << "\n";
  printMap(lossParams);

  std::map<std::string, double> initParams;
  std::string initType;
  BOOST_FOREACH(boost::property_tree::ptree::value_type const &v,
                init.get_child(""))
  {
    const boost::property_tree::ptree &attributes = v.second;
    if (v.first == "type")
    {
      initType = attributes.get_value<std::string>();
    }
    else
    {
      initParams[v.first] = attributes.get_value<double>();
    }
  }
  Log::Info << "\nInit details:\ntype : " << initType << "\n";
  printMap(initParams);
  std::map<std::string, double> optimizerDetails;
  std::string optimizerType;
  BOOST_FOREACH(boost::property_tree::ptree::value_type const &v,
                optimizer.get_child(""))
  {
    const boost::property_tree::ptree &attributes = v.second;
    if (v.first == "type")
    {
      optimizerType = attributes.get_value<std::string>();
    }
    else
    {
      optimizerDetails[v.first] = attributes.get_value<double>();
    }
  }
  Log::Info << "\nOptimizer details:\ntype : " << optimizerType << "\n";
  printMap(optimizerDetails);
  Log::Info << "\nNetwork details:\n\n";
  BOOST_FOREACH(boost::property_tree::ptree::value_type const &v,
                network.get_child(""))
  {
    const boost::property_tree::ptree &layerWhole = v.second;
    std::map<std::string, double> params;
    std::string layerType;
    BOOST_FOREACH(boost::property_tree::ptree::value_type const &v2,
                  layerWhole.get_child(""))
    {
      const boost::property_tree::ptree &layerInner = v2.second;
      std::string key = boost::erase_all_copy(v2.first, "_");
      boost::to_lower(key);
      if (key == "type")
      {
        layerType = layerInner.get_value<std::string>();
      }
      else if (key == "units")
      {
        params["insize"] = inSize;
        inSize = params["outsize"] = layerInner.get_value<double>();
      }
      else
      {
        if (key == "outsize")
          inSize = layerInner.get_value<double>();
        params[key] = layerInner.get_value<double>();
      }
    }
    Log::Info << "type : " << layerType << "\n";
    printMap(params);
    layers.push(getNetworkReference(layerType, params));
    Log::Info << "\n";
  }
  getInitType(initType, lossType, initParams, lossParams,
      optimizerType, optimizerDetails, layers, dataset);
}

boost::property_tree::ptree loadProperties(std::string& fileName,
                                           Dataset& dataset, double inSize)
{
  boost::property_tree::ptree pt;
  read_json(fileName, pt);
  traverseModel(pt, dataset, inSize);
  return pt;
}

int testParser()
{
  std::string fileName = "network3.json";
  arma::mat dataset2;
  data::Load("train.csv", dataset2, true);
  Log::Info << "Data loaded" << "\n\n";
  dataset2 = dataset2.submat(0, 1, dataset2.n_rows - 1, dataset2.n_cols - 1);
  arma::mat train, valid;
  data::Split(dataset2, train, valid, 0.1);
  arma::mat trainX = normalise(train.submat(1, 0, train.n_rows-1,
                               train.n_cols-1));
  arma::mat trainY = train.row(0) + 1;
  arma::mat validX = normalise(valid.submat(1, 0, valid.n_rows-1,
                               valid.n_cols-1));
  arma::mat validY = valid.row(0) + 1;
  Dataset dataset(trainX, trainY, validX, validY);
  loadProperties(fileName, dataset, trainX.n_rows);
  return 0;
}

#endif
