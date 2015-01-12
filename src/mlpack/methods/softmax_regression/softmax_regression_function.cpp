/**
 * @file softmax_regression_function.cpp
 * @author Siddharth Agrawal
 *
 * Implementation of function to be optimized for softmax regression.
 */
 #include "softmax_regression_function.hpp"

using namespace mlpack;
using namespace mlpack::regression;

SoftmaxRegressionFunction::SoftmaxRegressionFunction(const arma::mat& data,
                                                     const arma::vec& labels,
                                                     const size_t inputSize,
                                                     const size_t numClasses,
                                                     const double lambda) :
    data(data),
    labels(labels),
    inputSize(inputSize),
    numClasses(numClasses),
    lambda(lambda)
{
  // Intialize the parameters to suitable values.
  initialPoint = InitializeWeights();

  // Calculate the label matrix.
  GetGroundTruthMatrix(labels, groundTruth);
}

/**
 * Initializes parameter weights to random values taken from a scaled standard
 * normal distribution. The weights cannot be initialized to zero, as that will
 * lead to each class output being the same.
 */
const arma::mat SoftmaxRegressionFunction::InitializeWeights()
{
  // Initialize values to 0.005 * r. 'r' is a matrix of random values taken from
  // a Gaussian distribution with mean zero and variance one.
  arma::mat parameters;
  parameters.randn(numClasses, inputSize);
  parameters = 0.005 * parameters;

  return parameters;
}

/**
 * This is equivalent to applying the indicator function to the training
 * labels. The output is in the form of a matrix, which leads to simpler
 * calculations in the Evaluate() and Gradient() methods.
 */
void SoftmaxRegressionFunction::GetGroundTruthMatrix(const arma::vec& labels,
                                                     arma::sp_mat& groundTruth)
{
  // Calculate the ground truth matrix according to the labels passed. The
  // ground truth matrix is a matrix of dimensions 'numClasses * numExamples',
  // where each column contains a single entry of '1', marking the label
  // corresponding to that example.

  // Row pointers and column pointers corresponding to the entries.
  arma::uvec rowPointers(labels.n_elem);
  arma::uvec colPointers(labels.n_elem + 1);

  // Row pointers are the labels of the examples, and column pointers are the
  // number of cumulative entries made uptil that column.
  for(size_t i = 0; i < labels.n_elem; i++)
  {
    rowPointers(i) = labels(i, 0);
    colPointers(i+1) = i + 1;
  }

  // All entries are '1'.
  arma::vec values;
  values.ones(labels.n_elem);

  // Calculate the matrix.
  groundTruth = arma::sp_mat(rowPointers, colPointers, values, numClasses,
                             labels.n_elem);
}

/**
 * Evaluates the objective function given the parameters.
 */
double SoftmaxRegressionFunction::Evaluate(const arma::mat& parameters) const
{
  // The objective function is the negative log likelihood of the model
  // calculated over all the training examples. Mathematically it is as follows:
  // log likelihood = sum(1{y_i = j} * log(probability(j))) / m
  // The sum is over all 'i's and 'j's, where 'i' points to a training example
  // and 'j' points to a particular class. 1{x} is an indicator function whose
  // value is 1 only when 'x' is satisfied, otherwise it is 0.
  // 'm' is the number of training examples.
  // The cost also takes into account the regularization to control the
  // parameter weights.

  // Calculate the class probabilities for each training example. The
  // probabilities for each of the classes are given by:
  // p_j = exp(theta_j' * x_i) / sum(exp(theta_k' * x_i))
  // The sum is calculated over all the classes.
  // x_i is the input vector for a particular training example.
  // theta_j is the parameter vector associated with a particular class.
  arma::mat hypothesis, probabilities;

  hypothesis = arma::exp(parameters * data);
  probabilities = hypothesis / arma::repmat(arma::sum(hypothesis, 0),
                                            numClasses, 1);

  // Calculate the log likelihood and regularization terms.
  double logLikelihood, weightDecay, cost;

  logLikelihood = arma::accu(groundTruth % arma::log(probabilities)) /
      data.n_cols;
  weightDecay = 0.5 * lambda * arma::accu(parameters % parameters);

  // The cost is the sum of the negative log likelihood and the regularization
  // terms.
  cost = -logLikelihood + weightDecay;

  return cost;
}

/**
 * Calculates and stores the gradient values given a set of parameters.
 */
void SoftmaxRegressionFunction::Gradient(const arma::mat& parameters,
                                         arma::mat& gradient) const
{
  // Calculate the class probabilities for each training example. The
  // probabilities for each of the classes are given by:
  // p_j = exp(theta_j' * x_i) / sum(exp(theta_k' * x_i))
  // The sum is calculated over all the classes.
  // x_i is the input vector for a particular training example.
  // theta_j is the parameter vector associated with a particular class.
  arma::mat hypothesis, probabilities;

  hypothesis = arma::exp(parameters * data);
  probabilities = hypothesis / arma::repmat(arma::sum(hypothesis, 0),
                                            numClasses, 1);

  // Calculate the parameter gradients.
  gradient = (probabilities - groundTruth) * data.t() / data.n_cols +
      lambda * parameters;
}
