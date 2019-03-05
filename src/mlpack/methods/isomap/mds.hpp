/**
 * @file mds.hpp
 * @author Rishabh Ranjan
 * 
 * This file defines MDS class which implements classical Multidimensional
 * Scaling which is a dimensionality reduction algorithm which uses spectral
 * decomposition of dissimilarity matix.
 * This implementation is based on MATLAB's implementation of classical
 * multidimensional scaling (file - cmdscale.m).
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ISOMAP_MDS
#define MLPACK_METHODS_ISOMAP_MDS

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace isomap {

/**
 * This class implements classical Multidimensional Scaling.
 */
class MDS
{
 public:
  /**
   * Fucntion to implement classical Multidimensional Scaling on a given
   * dissimilarity matrix.
   * 
   * @param disMat -dissimilarity matrix (distance matrix)
   */
  void Apply(arma::mat& disMat)
  {
    // check to see if the incoming matrix is valid for cMDS
    IsValidMatrix(disMat);

    size_t n = disMat.n_rows;
    arma::mat temp(n, n);

    // Spectral decomposition

    arma::mat H = arma::eye<arma::mat>(n, n) - temp.fill((double) (1.0/n));
    arma::mat B = H * (-0.5 * (disMat % disMat) ) * H;
    arma::mat eigen_vecs;
    arma::vec eigen_vals;

    // guard eigen values against round-off errors
    arma::eig_sym(eigen_vals, eigen_vecs, (B + B.t())/2);

    arma::uvec index = arma::sort_index(eigen_vals, "descend");
    eigen_vals = arma::sort(eigen_vals, "descend");

    // keep only positive eigen values
    arma::uvec keep = arma::find(eigen_vals >
        arma::max(arma::abs(eigen_vals)) * (double) pow(DBL_EPSILON, 0.75));

    if (keep.is_empty())
    {
      // if none of the eigen values are positive, vector of zeros is returned
      disMat = arma::zeros<arma::vec>(n);
    }
    else
    {
      index = index(keep);
      disMat = 
      eigen_vecs.cols(index) * arma::diagmat(arma::pow(eigen_vals(keep), 0.5));
    }
  }

 private:
  /**
   * Function to check if the dissimilarity (distance) matrix provided is
   * valid for classical Multidimensional Scaling. Program terminates if it is
   * not.
   * 
   * @param disMat -dissimilarity (distance) matrix
   */
  void IsValidMatrix(arma::mat& disMat)
  {
    bool flag = 1;
    double minDiff = 10*DBL_EPSILON;
    // checking if matrix is a square matrix
    flag = disMat.is_square();
    if (flag)
    {
      // Checking for symmetric. could not find is_symmetric in armadillo
      flag = arma::approx_equal(disMat, disMat.t(), "absdiff", minDiff);
      if (flag)
      {
        // checking if all elements are non-zero
        flag = arma::all(vectorise(disMat) >= 0);
        if (!flag)
        {
          Log::Fatal << "All elements of the matrix must be non-zero (required "
                  << "for classical Multidimensional Scaling\n";
        }
      }
      else
      {
        Log::Fatal << "Matrix provided is not a symmetric matrix (required "
                  << "for classical Multidimensional Scaling\n";
      }
    }
    else
    {
      Log::Fatal << "Matrix provided is not a square matrix (required "
                  << "for classical Multidimensional Scaling\n";
    }
  }
};

} // namespace isomap&quot;
} // namepsace mlpack&quot;

#endif
