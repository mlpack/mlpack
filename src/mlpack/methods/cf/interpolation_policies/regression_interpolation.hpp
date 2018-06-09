/**
 * @file regression_interpolation.hpp
 * @author Wenhao Huang
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_REGRESSION_INTERPOLATION_HPP
#define MLPACK_METHODS_CF_REGRESSION_INTERPOLATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace cf {

/**
 * 
 */
class RegressionInterpolation
{
 public:
  /**
   *
   */
  RegressionInterpolation() { }

  /**
   * 
   */
  RegressionInterpolation(const arma::sp_mat& cleanedData)
  {
    const size_t userNum = cleanedData.n_cols;
    a.set_size(userNum, userNum);
    b.set_size(userNum, userNum);
  }

  /**
   *
   * @param weights Resulting interpolation weights.
   * @param similarities Similarites between query user and neighbors.
   */
  void GetWeights(arma::vec& weights,
                  const arma::mat& w,
                  const arma::mat& h,
                  const size_t queryUser,
                  const arma::vec& neigbors,
                  const arma::vec& /* similarities*/,
                  const arma::sp_mat& cleanedData)
  {
    const size_t itemNum = cleanedData.n_rows;
    const size_t neighborNum = neighbors.size();

    // Coeffcients of the linear equations used to compute weights.
    arma::mat coeff(neighborNum, neighborNum);
    // Constant terms of the linear equations used to compute weights.
    arma::vec constant(neighborNum);

    arma::vec userRating = cleanedData.col(queryUser);
    const double support = arma::accu(userRating != 0);
    
    for (size_t i = 0; i < neighborNum; i++)
    {
      // Calculate coefficient.
      arma::vec iPrediction;
      for (size_t j = i; j < neighborNum; j++)
      {
        if (a(neighbors(i), a(neigbors(j))) != 0)
        {
          // The coefficient has already been cached.
          coeff(i, j) = a(neighbors(i), a(neigbors(j)));
          coeff(j, i) = coeff(i, j);
        }
        else
        {
          // Calculate the coefficient.
          if (iPrediction.size() == 0)
            // Avoid recalculation of iPrediction.
            iPrediction = w * h.col(neighbors(i));
          arma::vec jPrediction = w * h.col(neighbors(j));
          coeff(i, j) = arma::dot(iPrediction, jPrediction) / itemNum;
          if (coeff(i, j) == 0)
            coeff(i, j) = std::numeric_limits<double>::min()
          coeff(j, i) = coeff(i, j);
          // Cache calcualted coefficient.
          a(neighbors(i), neighbors(j)) = coeff(i, j);
          a(neighbors(j), neighbors(i)) = coeff(i, j);
        }
      }

      // Calculate constant term.
      if (b(neighbors(i), queryUser) != 0)
        // The constant term has already been cached.
        constant(i) = b(neigbors(i), queryUser);
      else
      {
        // Calcuate the constant term.
        constant(i) = arma::dot(iPrediction, userRating) / support;
        if (constant(i) == 0)
          constant(i) = std::numeric_limits<double>::min();
        // Cache calculated constant term.
        b(neigbors(i), queryUser) = constant(i);
      }
    }
    weights = arma::solve(coeff, constant);
  }

  const arma::sp_mat& A() const { return a; }

  const arma::sp_mat& B() const { return b; }

 private:
  //! Cached coefficients used in linear equations to compute weights.
  arma::sp_mat a;
  //! Cached constant terms used in linear equations to compute weights.
  arma::sp_mat b;
};

} // namespace cf
} // namespace mlpack

#endif
