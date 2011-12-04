/**
 * @file gmm_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of template-based GMM methods.
 */
#ifndef __MLPACK_METHODS_GMM_GMM_IMPL_HPP
#define __MLPACK_METHODS_GMM_GMM_IMPL_HPP

// In case it hasn't already been included.
#include "gmm.hpp"

namespace mlpack {
namespace gmm {

template<typename ClusteringType>
void GMM::InitialClustering(const ClusteringType& clusterer,
                            const arma::mat& data,
                            std::vector<arma::vec>& meansOut,
                            std::vector<arma::mat>& covarsOut,
                            arma::vec& weightsOut) const
{
  meansOut.resize(gaussians, arma::vec(dimensionality));
  covarsOut.resize(gaussians, arma::mat(dimensionality, dimensionality));
  weightsOut.set_size(gaussians);

  // Assignments from clustering.
  arma::Col<size_t> assignments;

  // Run clustering algorithm.
  clusterer.Cluster(data, gaussians, assignments);

  // Now calculate the means, covariances, and weights.
  weightsOut.zeros();
  for (size_t i = 0; i < gaussians; i++)
  {
    meansOut[i].zeros();
    covarsOut[i].zeros();
  }

  // From the assignments, generate our means, covariances, and weights.
  for (size_t i = 0; i < data.n_cols; i++)
  {
    size_t cluster = assignments[i];

    // Add this to the relevant mean.
    meansOut[cluster] += data.col(i);

    // Add this to the relevant covariance.
    covarsOut[cluster] += data.col(i) * trans(data.col(i));

    // Now add one to the weights (we will normalize).
    weightsOut[cluster]++;
  }

  // Now normalize the mean and covariance.
  for (size_t i = 0; i < gaussians; i++)
  {
    covarsOut[i] -= meansOut[i] * trans(meansOut[i]) /
        weightsOut[i];

    meansOut[i] /= weightsOut[i];

    covarsOut[i] /= (weightsOut[i] > 1) ? weightsOut[i] : 1;
  }

  // Finally, normalize weights.
  weightsOut /= accu(weightsOut);
}

}; // namespace gmm
}; // namespace mlpack

#endif
