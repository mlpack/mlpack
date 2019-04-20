/**
 * @file cmds.hpp
 * @author Rishabh Ranjan
 * 
 * This file defines MDS class which implements classical Multidimensional
 * Scaling which tries to find a set a d-dimensional points whose interpoint
 * distances correspond to some extent to the given dissimilarity matrix, or.
 * perform non-linear dimensionnality reduction on a given dataset.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_CMDS
#define MLPACK_METHODS_CMDS

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace cmds {

/**
 * This class implements classical Multidimensional Scaling.
 */
class Cmds
{
 public:
  /**
   * Fucntion to implement classical Multidimensional Scaling on a given
   * dissimilarity matrix.
   *
   * @param cd Whether to calculate dissimilarity matrix or it is the input.
   * @param d Number of dimesions that output should have.
   * @param data Dissimilarity matrix to work on.
   */
  void Apply(bool cd,
             size_t d,
             arma::mat& data)
  {
    // Changing input dataset to dissimilarity matrix if required
    if (cd)
      CalcDissimilarityMatrix(data);
    size_t n = data.n_rows;

    // Spectral decomposition

    // Gram matrix for a mean-centered configuration with interpoint distances
    // given by dissimilarity matrix.
    arma::mat H = arma::eye<arma::mat>(n, n) - (double) (1.0/n);
    arma::mat B = H * (-0.5 * (data % data) ) * H;

    arma::mat eigenVecs;
    arma::vec eigenVals;

    arma::eig_sym(eigenVals, eigenVecs, B);

    // Keep only positive eigen values.
    arma::uvec keep = arma::find(eigenVals >
        arma::max(arma::abs(eigenVals)) * (double) pow(DBL_EPSILON, 0.75));

    if (keep.is_empty())
    {
      // If none of the eigen values are positive, vector of zeros is returned.
      data = arma::zeros<arma::vec>(n);
    }
    else
    {
      data = eigenVecs.cols(keep) *
          arma::diagmat(arma::pow(eigenVals(keep), 0.5));

      // Fliping rows so that initial columns correspond to higher eigen
      // values.
      data = arma::fliplr(data);
      if (d > data.n_cols)
        Log::Fatal << "Desired number of dimensions is more than what "
            << "the points can be represented with in Euclidean space ("
            << data.n_cols << ")\n"; 
      if (d > 0)
        data = data.cols(0, d-1);
      data = data.t();
    }
  }

 private:
  /**
   * Function to calculate dissimilarity matrix from the given input dataset.
   * 
   * @param data Input dataset.
   */
  void CalcDissimilarityMatrix(arma:: mat& data)
  {
    arma::mat disMat(data.n_cols, data.n_cols);
    for (size_t i = 0; i < data.n_cols; i++)
    {
      disMat(i, i) = 0;
      for (size_t j = i+1; j < data.n_cols; j++)
      {
        disMat(i, j) = metric::EuclideanDistance().Evaluate(data.col(i),
                                                            data.col(j));
        disMat(j, i) = disMat(i, j);
      }
    }
    data = disMat;
  }
};

} // namespace cmds
} // namespace mlpack

#endif
