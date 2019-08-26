/**
 * @file frechet_distance.hpp
 * @author Toshal Agrawal
 *
 * Definition of Frechet Inception Distance for Generative Adversarial Networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_GAN_METRICS_FRECHET_DISTANCE_HPP
#define MLPACK_METHODS_ANN_GAN_METRICS_FRECHET_DISTANCE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /* Artificial Neural Network */ {

/**
* Function that computes Frechet Inception Distance for a set of images
* produced by a GAN.
*
* For more information, see the following.
*
* @code
* @article{
*   author    = {Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard
*                Nessler, Sepp Hochreiter},
*   title     = {GANs Trained by a Two Time-Scale Update Rule Converge to a
*                Local Nash Equilibrium},
*   year      = {2017},
*   url       = {https://arxiv.org/abs/1706.08500},
*   eprint    = {1706.08500},
* }
* @endcode
*/
class FrechetDistance
{
 public:
  /**
  * Function to calculate Frechet Distance for the given distribution.
  *
  * @param p The generating model data.
  * @param pw The real world data.
  */
  double Evaluate(const arma::mat& p, const arma::mat& pw)
  {
    // Calculate norm of mean.
    double firstTerm = arma::norm(arma::mean(p, 1) - arma::mean(pw, 1), 2);

    // p and pw will be nDims * nPoints matrix. Hence taking transpose to find
    // covariance.
    const arma::mat c = arma::cov(p.t());
    const arma::mat cW = arma::cov(pw.t());

    return firstTerm + arma::trace(c + cW - 2 * arma::sqrt(c * trans(cW)));
  }
};

} // namespace ann
} // namespace mlpack

#endif
