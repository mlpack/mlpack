/**
 * @file cyclegan.hpp
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_CYCLEGAN_HPP
#define MLPACK_METHODS_ANN_GAN_CYCLEGAN_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/visitor/output_parameter_visitor.hpp>
#include <mlpack/methods/ann/visitor/reset_visitor.hpp>
#include <mlpack/methods/ann/visitor/weight_size_visitor.hpp>
#include <mlpack/methods/ann/visitor/weight_set_visitor.hpp>


namespace mlpack {
namespace ann /** Artificial Neural Network. **/ {

/**
 * The implementation of the CycleGAN module. Generative Adversarial Networks
 * (GANs) are a class of artificial intelligence algorithms used in unsupervised
 * machine learning, implemented by a system of two neural networks contesting
 * with each other in a zero-sum game framework. This type of GAN is readily
 * used for image-to-image translation, a class of vision and graphics problems
 * where the goal is to learn the mapping between an input image and an output
 * image using a training set of aligned image pairs, in the case where paired
 * data might not available in abundance.
 *
 * For more information, see the following paper:
 *
 * @code
 * @article{Zhu17,
 *   author    = {Jun-Yan Zhu, Taesung Park, Phillip Isola and Alexei A. Efros},
 *   title     = {Unpaired Image-to-Image Translation using Cycle-Consistent
                  Adversarial Networks},
 *   year      = {2017},
 *   url       = {https://arxiv.org/abs/1703.10593},
 *   eprint    = {1703.10593},
 * }
 * @endcode
 *
 * @tparam Model The class type of Generator and Discriminator.
 * @tparam InitializationRuleType Type of Initializer.
 */
template<
  typename Model,
  typename InitializationRuleType
>
class CycleGAN
{
 public:
  /**
   * Constructor for CycleGAN class.
   *
   * @param trainDataX The real data.
   * @param trainDataY The real data.
   * @param generatorX Generator network.
   * @param generatorY Generator network.
   * @param discriminatorX Discriminator network.
   * @param discriminatorY Discriminator network.
   * @param batchSize Batch size to be used for training.
   * @param generatorUpdateStep Number of steps to train Discriminator
   *                            before updating Generator.
   * @param preTrainSize Number of pre-training steps of Discriminator.
   * @param lambda Regularization term for cyclic loss.
   */
  CycleGAN(arma::mat& trainDataX,
           arma::mat& trainDataY,
           Model generatorX,
           Model generatorY,
           Model discriminatorX,
           Model discriminatorY,
           InitializationRuleType& initializeRule,
           const size_t batchSize,
           const size_t generatorUpdateStep,
           const size_t preTrainSize,
           const double lambda = 10.0,
           const double multiplier = 10.0);

  //! Copy constructor.
  CycleGAN(const CycleGAN&);

  //! Move constructor.
  CycleGAN(CycleGAN&&);

  // Reset function.
  void Reset();

  // Train function.
  template<typename OptimizerType>
  void Train(OptimizerType& Optimizer);

  /**
   * Evaluate function for the CycleGAN.
   * This function gives the performance of the CycleGAN on the current input.
   *
   * @param parameters The parameters of the network.
   * @param i Index of the current input.
   * @param batchSize Variable to store the present number of inputs.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t i,
                  const size_t batchSize);

  /**
   * EvaluateWithGradient function for the CycleGAN.
   * This function gives the performance of the CycleGAN on the current input,
   * while updating Gradients.
   *
   * @param parameters The parameters of the network.
   * @param i Index of the current input.
   * @param gradient Variable to store the present gradient.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename GradType>
  double EvaluateWithGradient(const arma::mat& parameters,
                              const size_t i,
                              GradType& gradient,
                              const size_t batchSize);

  /**
   * Gradient function for CycleGAN.
   * This function passes the gradient based on which network is being
   * trained, i.e., Generator or Discriminator.
   * 
   * @param parameters present parameters of the network.
   * @param i Index of the predictors.
   * @param gradient Variable to store the present gradient.
   * @param batchSize Variable to store the present number of inputs.
   */
  void Gradient(const arma::mat& parameters,
                const size_t i,
                arma::mat& gradient,
                const size_t batchSize);

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  /**
   * This function does a forward pass through the GAN network.
   *
   * @param input Image from either domain.
   * @param xtoy Indicator of source and target domains.
   */
  void Forward(arma::mat&& input, const bool xtoy);

  /**
   * This function predicts the output of the network on the given input.
   *
   * @param input The input the Discriminator network.
   * @param output Result of the Discriminator network.
   * @param xtoy Indicator of source and target domains.
   */
  void Predict(arma::mat&& input, arma::mat& output, const bool xtoy = true);

  //! Return the parameters of the network.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the parameters of the network.
  arma::mat& Parameters() { return parameter; }

  //! Return the generatorX of the CycleGAN.
  const Model& GeneratorX() const { return generatorX; }
  //! Modify the generatorX of the CycleGAN.
  Model& GeneratorX() { return generatorX; }
  //! Return the generatorY of the CycleGAN.
  const Model& GeneratorY() const { return generatorY; }
  //! Modify the generatorY of the CycleGAN.
  Model& GeneratorY() { return generatorY; }
  //! Return the discriminatorX of the CycleGAN.
  const Model& DiscriminatorX() const { return discriminatorX; }
  //! Modify the discriminatorX of the CycleGAN.
  Model& DiscriminatorX() { return discriminatorX; }
  //! Return the discriminatorY of the CycleGAN.
  const Model& DiscriminatorY() const { return discriminatorY; }
  //! Modify the discriminatorY of the CycleGAN.
  Model& DiscriminatorY() { return discriminatorY; }

  //! Return the number of separable functions (the number of predictor points
  //! in domain X).
  size_t NumFunctionsX() const { return numFunctionsX; }
  //! Return the number of separable functions (the number of predictor points
  //! in domain Y).
  size_t NumFunctionsY() const { return numFunctionsY; }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally stored parameter for training data + generated data.
  arma::mat predictorsX;
  //! Locally stored parameter for training data + generated data.
  arma::mat predictorsY;
  //! Locally stored parameters of the network.
  arma::mat parameter;
  //! Locally stored Generator network.
  Model generatorX;
  //! Locally stored Generator network.
  Model generatorY;
  //! Locally stored Discriminator network.
  Model discriminatorX;
  //! Locally stored Discriminator network.
  Model discriminatorY;
  //! Locally stored Initializer.
  InitializationRuleType initializeRule;
  //! Locally stored number of data points in domain X.
  size_t numFunctionsX;
  //! Locally stored number of data points in domain Y.
  size_t numFunctionsY;
  //! Locally stored batch size parameter.
  size_t batchSize;
  //! Locally stored number of iterations of X that have been completed.
  size_t counterX;
  //! Locally stored number of iterations of Y that have been completed.
  size_t counterY;
  //! Locally stored batch number which is being processed.
  size_t currentBatch;
  //! Locally stored number of training step before Generator is trained.
  size_t generatorUpdateStep;
  //! Locally stored number of pre-train step for Discriminator.
  size_t preTrainSize;
  //! Locally stored regularization term for cyclic loss.
  double lambda;
  //! Locally stored learning rate ratio for Generator network.
  double multiplier;
  //! Locally stored reset parameter.
  bool reset;
  //! Locally stored delta visitor.
  DeltaVisitor deltaVisitor;
  //! Locally stored responses for domain X.
  arma::mat responsesX;
  //! Locally stored responses for domain Y.
  arma::mat responsesY;
  //! Locally stored current input.
  arma::mat currentInput;
  //! Locally stored current target.
  arma::mat currentTarget;
  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;
  //! Locally-stored weight size visitor.
  WeightSizeVisitor weightSizeVisitor;
  //! Locally-stored reset visitor.
  ResetVisitor resetVisitor;
  //! Locally stored gradient parameters.
  arma::mat gradient;
  //! Locally stored gradient for Discriminator.
  arma::mat gradientDiscriminatorX;
  //! Locally stored gradient for Generator.
  arma::mat gradientGeneratorX;
  //! Locally stored gradient for Discriminator.
  arma::mat gradientDiscriminatorY;
  //! Locally stored gradient for Generator.
  arma::mat gradientGeneratorY;
  //! Locally stored gradient for data generated from the predictorsY.
  arma::mat generatedGradientDX;
  //! Locally stored gradient for data generated from the predictorsX.
  arma::mat generatedGradientDY;
  //! Locally stored gradient for regenerated X data.
  arma::mat generatedGradientGX;
  //! Locally stored gradient for regenerated Y data.
  arma::mat generatedGradientGY;
  //! Locally stored output of the Generator network.
  arma::mat ganOutput;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "cyclegan_impl.hpp"

#endif
