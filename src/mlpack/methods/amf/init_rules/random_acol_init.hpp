/**
 * @file methods/amf/init_rules/random_acol_init.hpp
 * @author Mohan Rajendran
 *
 * Initialization rule for Alternating Matrix Factorization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMF_RANDOM_ACOL_INIT_HPP
#define MLPACK_METHODS_LMF_RANDOM_ACOL_INIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

namespace mlpack {

/**
 * This class initializes the W matrix of the AMF algorithm by averaging p
 * randomly chosen columns of V.  In this case, p is a template parameter.  H is
 * then filled using a uniform distribution in the range [0, 1].
 *
 * This simple initialization is the "random Acol initialization" found in the
 * following paper:
 *
 * @code
 * @techreport{langville2014algorithms,
 *   title = {Algorithms, Initializations, and Convergence for the Nonnegative
 *       Matrix Factorization},
 *   author = {Langville, A.N. and Meyer, C.D. and Albright, R. and Cox, J. and
 *       Duling, D.},
 *   year = {2014},
 *   institution = {NCSU Technical Report Math 81706}
 * }
 * @endcode
 *
 * @tparam columnsToAverage The number of random columns to average for each
 *     column of W.
 */
template<size_t columnsToAverage = 5>
class RandomAcolInitialization
{
 public:
  // Empty constructor required for the InitializeRule template
  RandomAcolInitialization()
  { }

  template<typename MatType, typename WHMatType>
  inline static void Initialize(const MatType& V,
                                const size_t r,
                                WHMatType& W,
                                WHMatType& H)
  {
    const size_t n = V.n_rows;
    const size_t m = V.n_cols;

    if (columnsToAverage > m)
    {
      Log::Warn << "Number of random columns (columnsToAverage) is more than "
          << "the number of columns available in the V matrix; weird results "
          << "may ensue!" << std::endl;
    }

    W.zeros(n, r);

    // Initialize W matrix with random columns.
    for (size_t col = 0; col < r; col++)
    {
      for (size_t randCol = 0; randCol < columnsToAverage; randCol++)
      {
        W.col(col) += V.col(RandInt(0, m));
      }
    }

    // Now divide by p.
    W /= columnsToAverage;

    // Initialize H to random values.
    H.randu(r, m);
  }

  /**
   * Initialize the matrix W or H only.
   *
   * @param V Input matrix.
   * @param r Rank of matrix.
   * @param M W or H matrix.
   * @param whichMatrix If true, initialize W. Otherwise, initialize H.
   */
  template<typename MatType, typename WHMatType>
  inline static void InitializeOne(const MatType& V,
                                   const size_t r,
                                   WHMatType& M,
                                   const bool whichMatrix = true)
  {
    const size_t n = V.n_rows;
    const size_t m = V.n_cols;

    if (columnsToAverage > m)
    {
      Log::Warn << "Number of random columns (columnsToAverage) is more than "
          << "the number of columns available in the V matrix; weird results "
          << "may ensue!" << std::endl;
    }

    if (whichMatrix)
    {
      // Initialize W matrix.
      M.zeros(n, r);

      // Initialize W matrix with random columns.
      for (size_t col = 0; col < r; col++)
      {
        for (size_t randCol = 0; randCol < columnsToAverage; randCol++)
        {
          M.col(col) += V.col(RandInt(0, m));
        }
      }

      // Now divide by p.
      M /= columnsToAverage;
    }
    else
    {
      // Initialize H to random values.
      M.randu(r, m);
    }
  }

  //! Serialize the object (in this case, there is nothing to serialize).
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */) { }
};

} // namespace mlpack

#endif
