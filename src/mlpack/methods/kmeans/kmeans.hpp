/**
 * @file kmeans.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * K-Means clustering.
 */
#ifndef __MLPACK_METHODS_KMEANS_KMEANS_HPP
#define __MLPACK_METHODS_KMEANS_KMEANS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kmeans {

/**
 * This function computes the k-means of the data and stores the calculated
 * means and covariances in the std::vector of vectors and matrices passed to
 * it.  It sets the weights uniformly.
 *
 * This function is used to obtain a starting point for the optimization.
 */
void KMeans(const arma::mat& data,
            const size_t value_of_k,
            std::vector<arma::vec>& means,
            std::vector<arma::mat>& covars,
            arma::vec& weights);

}; // namespace gmm
}; // namespace mlpack

#endif // __MLPACK_METHODS_MOG_KMEANS_HPP
