/**
 * @file methods/cf/interpolation_policies/regression_interpolation.hpp
 * @author Wenhao Huang
 *
 * Definition of RegressionInterpolation class.
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

/**
 * Implementation of regression-based interpolation method. Predicting a user's
 * rating \f$ r_{iu} \f$ by it's neighbors' ratings can be regarded as solving
 * linear regression of \f$ r_{iu} \f$ on \f$ r_{iv} \f$, where v are u's
 * neighbors.
 *
 * An example of how to use RegressionInterpolation in CF is shown below:
 *
 * @code
 * extern arma::mat data; // data is a (user, item, rating) table.
 * // Users for whom recommendations are generated.
 * extern arma::Col<size_t> users;
 * arma::Mat<size_t> recommendations; // Resulting recommendations.
 *
 * CFType<> cf(data);
 *
 * // Generate 10 recommendations for all users.
 * cf.template GetRecommendations<
 *     EuclideanSearch,
 *     RegressionInterpolation>(10, recommendations);
 * @endcode
 *
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings{bell2007improved,
 *  title={Improved neighborhood-based collaborative filtering},
 *  author={Bell, Robert M and Koren, Yehuda},
 *  booktitle={KDD cup and workshop at the 13th ACM SIGKDD international
 *      conference on knowledge discovery and data mining},
 *  pages={7--14},
 *  year={2007},
 *  organization={Citeseer}
 * }
 * @endcode
 */
class RegressionInterpolation
{
 public:
  /**
   * Empty Constructor.
   */
  RegressionInterpolation() { }

  /**
   * Use cleanedData to perform necessary preprocessing.
   *
   * @param cleanedData Sparse rating matrix.
   */
  RegressionInterpolation(const arma::sp_mat& cleanedData)
  {
    const size_t userNum = cleanedData.n_cols;
    a.set_size(userNum, userNum);
    b.set_size(userNum, userNum);
  }

  /**
   * The regression-based interpolation problem can be solved by a linear
   * system of equations. This method first calculates the coefficients and
   * constant terms for the equations and then solve the equations. The
   * solution of the linear system of equations is the resulting interpolation
   * weights (the first parameter). After getting the weights, CF algorithm
   * multiplies each neighbor's rating by its corresponding weight and sums
   * them to get predicted rating.
   *
   * @param weights Resulting interpolation weights. The size of weights should
   *     be set to the number of neighbors before calling GetWeights().
   * @param decomposition Decomposition object.
   * @param queryUser Queried user.
   * @param neighbors Neighbors of queried user.
   * @param * (similarities) Similarities between query user and neighbors.
   * @param cleanedData Sparse rating matrix.
   */
  template <typename VectorType,
            typename DecompositionPolicy>
  void GetWeights(VectorType&& weights,
                  const DecompositionPolicy& decomposition,
                  const size_t queryUser,
                  const arma::Col<size_t>& neighbors,
                  const arma::vec& /* similarities*/,
                  const arma::sp_mat& cleanedData)
  {
    if (weights.n_elem != neighbors.n_elem)
    {
      Log::Fatal << "The size of the first parameter (weights) should "
          << "be set to the number of neighbors before calling GetWeights()."
          << std::endl;
    }

    const arma::mat& w = decomposition.W();
    const arma::mat& h = decomposition.H();
    const size_t itemNum = cleanedData.n_rows;
    const size_t neighborNum = neighbors.size();

    // Coeffcients of the linear equations used to compute weights.
    arma::mat coeff(neighborNum, neighborNum);
    // Constant terms of the linear equations used to compute weights.
    arma::vec constant(neighborNum);

    arma::vec userRating(cleanedData.col(queryUser));
    const size_t support = accu(userRating != 0);

    // If user has no rating at all, average interpolation is used.
    if (support == 0)
    {
      weights.fill(1.0 / neighbors.n_elem);
      return;
    }

    for (size_t i = 0; i < neighborNum; ++i)
    {
      // Calculate coefficient.
      arma::vec iPrediction;
      for (size_t j = i; j < neighborNum; ++j)
      {
        if (a(neighbors(i), neighbors(j)) != 0)
        {
          // The coefficient has already been cached.
          coeff(i, j) = a(neighbors(i), neighbors(j));
          coeff(j, i) = coeff(i, j);
        }
        else
        {
          // Calculate the coefficient.
          if (iPrediction.size() == 0)
            // Avoid recalculation of iPrediction.
            iPrediction = w * h.col(neighbors(i));
          arma::vec jPrediction = w * h.col(neighbors(j));
          coeff(i, j) = dot(iPrediction, jPrediction) / itemNum;
          if (coeff(i, j) == 0)
            coeff(i, j) = std::numeric_limits<double>::min();
          coeff(j, i) = coeff(i, j);
          // Cache calcualted coefficient.
          a(neighbors(i), neighbors(j)) = coeff(i, j);
          a(neighbors(j), neighbors(i)) = coeff(i, j);
        }
      }

      // Calculate constant terms.
      if (b(neighbors(i), queryUser) != 0)
        // The constant term has already been cached.
        constant(i) = b(neighbors(i), queryUser);
      else
      {
        // Calcuate the constant term.
        if (iPrediction.size() == 0)
            // Avoid recalculation of iPrediction.
            iPrediction = w * h.col(neighbors(i));
        constant(i) = dot(iPrediction, userRating) / support;
        if (constant(i) == 0)
          constant(i) = std::numeric_limits<double>::min();
        // Cache calculated constant term.
        b(neighbors(i), queryUser) = constant(i);
      }
    }
    weights = arma::solve(coeff, constant);
  }

 private:
  //! Cached coefficients used in linear equations to compute weights.
  arma::sp_mat a;
  //! Cached constant terms used in linear equations to compute weights.
  arma::sp_mat b;
};

} // namespace mlpack

#endif
