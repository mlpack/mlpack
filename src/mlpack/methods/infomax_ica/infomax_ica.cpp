/**
 * @file infomax_ica.cpp
 * @author Chip Mappus
 *
 * Methods for InfomaxICA.
 *
 * @see infomax_ica.h
 */
#include <mlpack/core.hpp>

#include "infomax_ica.hpp"

namespace mlpack {
namespace infomax_ica {

PARAM(double, "lambda", "The learning rate.", "info", .001,
  false);
PARAM_INT_REQ("B", "Infomax data window size.", "info");
PARAM(double, "epsilon", "Infomax algorithm stop threshold.", "info", .001,
  false);
PARAM_MODULE("info",
  "This performs ICO decomposition on a given dataset using the Infomax method");

/**
 * Dummy constructor
 */
InfomaxICA::InfomaxICA() { }

InfomaxICA::InfomaxICA(double lambda, size_t b, double epsilon) :
    lambda_(lambda),
    b_(b),
    epsilon_(epsilon) { }

/**
 * Sphere the data, apply ica. This is the first function to call
 * after initializing the variables.
 */
void InfomaxICA::applyICA(const arma::mat& dataset)
{
  double current_cos = DBL_MAX;
  w_.set_size(b_, b_);
  data_ = dataset; // Copy the matrix?  Stupid!

  if (b_ < data_.n_cols)
  {
    sphere(data_);

    // initial estimate for w is Id
    arma::mat i1;
    i1.eye(data_.n_rows, data_.n_rows);
    w_ = i1; // Another copy?  Stupid!
    while (epsilon_ <= current_cos)
    {
      arma::mat w_prev = w_;
      evaluateICA();
      current_cos = w_delta(w_prev, w_);
    }
  }
  else
    Log::Fatal << "Window size must be less than number of instances."
        << std::endl;
}

/**
 * Run infomax. Call this after initialization and applyICA.
 */
void InfomaxICA::evaluateICA()
{
  arma::mat BI;
  BI.eye(w_.n_rows, w_.n_rows);
  BI *= b_;

  // intermediate calculation variables
  arma::mat icv(w_.n_rows, b_);
  arma::mat icv2(w_.n_rows, w_.n_cols);
  arma::mat icv4(w_.n_rows, w_.n_cols);

  for (size_t i = 0; i < data_.n_cols; i += b_)
  {
    if ((i + b_) < data_.n_cols)
    {
      // This is not tested.
      icv = -2.0 * arma::pow(arma::exp(
          -1 * (w_ * data_.cols(i, i + b_))) + 1, -1) + 1;
      icv2 = icv * arma::trans(data_.cols(i, i + b_));
      icv4 = lambda_ * (icv2 + BI);
      icv2 = icv4 * w_;
      w_ += icv2;
    }
  }
}

// sphereing functions
/**
 * Sphere the input data.
 */
void InfomaxICA::sphere(arma::mat& data)
{
  arma::mat sample_covariance = sampleCovariance(data);
  arma::mat wz = sqrtm(sample_covariance);
  arma::mat data_sub_means = subMeans(data);

  arma::mat wz_inverse = inv(wz);

  // Not tested.
  wz_inverse *= 2.0;
  data = wz_inverse * data_sub_means;
}

// Covariance matrix.
arma::mat InfomaxICA::sampleCovariance(const arma::mat& m)
{
  // Not tested.
  arma::mat ttm = subMeans(m);
  arma::mat wm = trans(ttm);

  arma::mat twm(ttm.n_rows, ttm.n_rows);
  arma::mat output(ttm.n_rows, ttm.n_rows);
  output.zeros();

  arma::mat tttm(wm);

  tttm *= (1 / (double) (ttm.n_cols - 1));
  twm = trans(wm) * tttm;
  output = twm;

  return output;
}

arma::mat InfomaxICA::subMeans(const arma::mat& m)
{
  arma::mat output(m);
  arma::vec row_means = rowMean(output);

  for (size_t j = 0; j < output.n_cols; j++)
    output.col(j) -= row_means;

  return output;
}

/**
 * Compute the sample mean of a column
 */
arma::vec InfomaxICA::rowMean(const arma::mat& m)
{
  arma::vec row_means(m.n_rows);
  row_means.zeros();

  for (size_t j = 0; j < m.n_cols; j++)
    row_means += m.col(j);

  row_means /= (double) m.n_cols;

  return row_means;
}

/**
 * Matrix square root using Cholesky decomposition method.  Assumes the input
 * matrix is square.
 */
arma::mat InfomaxICA::sqrtm(const arma::mat& m)
{
  arma::mat output(m.n_rows, m.n_cols);
  arma::mat chol;

  if (arma::chol(chol, m))
  {
    arma::mat u, v;
    arma::vec s;

    if (arma::svd(u, s, v, trans(chol)))
    {
      arma::mat S(s.n_elem, s.n_elem);
      S.zeros();
      S.diag() = s;

      arma::mat tm1 = u * S;
      output = tm1 * trans(u);
    }
    else
      Log::Warn << "InfomaxICA sqrtm: SVD failed." << std::endl;
  }
  else
    Log::Warn << "InfomaxICA sqrtm: Cholesky decomposition failed." << std::endl;

  return output;
}

// Compare w estimates for convergence
double InfomaxICA::w_delta(const arma::mat& w_prev, const arma::mat& w_pres)
{
  arma::mat temp = w_pres - w_prev;
  arma::vec delta = reshape(temp, temp.n_rows * temp.n_cols, 1);
  double delta_dot = arma::dot(delta, delta);

  Log::Info << "w change: " << delta_dot << std::endl;
  return delta_dot;
}

/**
 * Return the current unmixing matrix estimate. Requires a reference
 * to an uninitialized matrix.
 */
void InfomaxICA::getUnmixing(arma::mat& w)
{
  w = w_;
}

/**
 * Return the source estimates, S. S is a reference to an
 * uninitialized matrix.
 */
void InfomaxICA::getSources(const arma::mat& dataset, arma::mat& s)
{
  s = w_ * dataset;
}

void InfomaxICA::setLambda(const double lambda)
{
  lambda_ = lambda;
}

void InfomaxICA::setB(const size_t b)
{
  b_ = b;
}

void InfomaxICA::setEpsilon(const double epsilon)
{
  epsilon_ = epsilon;
}

}; // namespace fastica
}; // namespace mlpack
