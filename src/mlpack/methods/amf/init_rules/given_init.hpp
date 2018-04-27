/**
 * @file given_initialization.hpp
 * @author Ryan Curtin
 *
 * Initialization rule for alternating matrix factorization (AMF). This simple
 * initialization is performed by assigning a given matrix to W and/or H.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AMF_INIT_RULES_GIVEN_INIT_HPP
#define MLPACK_METHODS_AMF_INIT_RULES_GIVEN_INIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace amf {

/**
 * This initialization rule for AMF simply fills the W and/or H matrices with
 * the matrices given to the constructor of this object. If only one initial
 * matrix is given, the other matrix will be initialized with random uniform 
 * noise. Note that this object does not use std::move() during Initialize()
 * method, so it can be reused for multiple AMF objects, but will incur copies
 * of the W and H matrices.
 */
class GivenInitialization
{
 public:
  // Empty constructor required for the InitializeRule template.
  GivenInitialization() : wIsGiven(false), hIsGiven(false) { }

  // Initialize the GivenInitialization object with the given matrices.
  GivenInitialization(const arma::mat& w, const arma::mat& h) :
    w(w), h(h), wIsGiven(true), hIsGiven(true) { }

  // Initialize the GivenInitialization object, taking control of the given
  // matrices.
  GivenInitialization(const arma::mat&& w, const arma::mat&& h) :
    w(std::move(w)),
    h(std::move(h)),
    wIsGiven(true),
    hIsGiven(true)
  { }

  // Initialize either H or W with the given matrix. The other matrix will
  // be initialized with random uniform noise in Initialize().
  GivenInitialization(const char whichMatrix, const arma::mat& m)
  {
    if (whichMatrix == 'W' || whichMatrix == 'w')
    {
      w = m;
      wIsGiven = true;
      hIsGiven = false;
    }
    else if (whichMatrix == 'H' || whichMatrix == 'h')
    {
      h = m;
      wIsGiven = false;
      hIsGiven = true;
    }
    else
    {
      Log::Fatal << "Specify either 'H' or 'W' when creating "
          "GivenInitialization object!" << std::endl;
    }
  }

  // Initialize either H or W, taking control of the given matrix. The other
  // matrix will be initialized with random uniform noise in Initialize().
  GivenInitialization(const char whichMatrix, const arma::mat&& m)
  {
    if (whichMatrix == 'W' || whichMatrix == 'w')
    {
      w = std::move(m);
      wIsGiven = true;
      hIsGiven = false;
    }
    else if (whichMatrix == 'H' || whichMatrix == 'h')
    {
      h = std::move(m);
      wIsGiven = false;
      hIsGiven = true;
    }
    else
    {
      Log::Fatal << "Specify either 'H' or 'W' when creating "
          "GivenInitialization object!" << std::endl;
    }
  }

  /**
   * Initialize H and W to the given matrices.
   *
   * @param V Input matrix.
   * @param r Rank of decomposition.
   * @param W W matrix, to be initialized to given matrix.
   * @param H H matrix, to be initialized to given matrix.
   */
  template<typename MatType>
  inline void Initialize(const MatType& V,
                         const size_t r,
                         arma::mat& W,
                         arma::mat& H)
  {
    // Make sure the initial W, H matrices have correct size.
    if (w.n_rows != V.n_rows && wIsGiven)
    {
      Log::Fatal << "The number of rows in given W (" <<  w.n_rows
          << ") doesn't equal the number of rows in V (" << V.n_rows
          << ") !" << std::endl;
    }
    if (w.n_cols != r && wIsGiven)
    {
      Log::Fatal << "The number of columns in given W (" <<  w.n_cols
          << ") doesn't equal the rank of factorization (" << r
          << ") !" << std::endl;
    }
    if (h.n_cols != V.n_cols && hIsGiven)
    {
      Log::Fatal << "The number of columns in given H (" <<  h.n_cols
          << ") doesn't equal the number of columns in V (" << V.n_cols
          << ") !" << std::endl;
    }
    if (h.n_rows != r && hIsGiven)
    {
      Log::Fatal << "The number of rows in given H (" <<  h.n_rows
          << ") doesn't equal the rank of factorization (" << r
          << ") !"<< std::endl;
    }

    // Initialize to the given matrices.
    if (wIsGiven)
      W = w;
    else
      W.randu(V.n_rows, r);
    if (hIsGiven)
      H = h;
    else
      H.randu(r, V.n_cols);
  }

  //! Serialize the object (in this case, there is nothing to serialize).
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(w);
    ar & BOOST_SERIALIZATION_NVP(h);
    ar & BOOST_SERIALIZATION_NVP(wIsGiven);
    ar & BOOST_SERIALIZATION_NVP(hIsGiven);
  }

 private:
  //! The W matrix for initialization.
  arma::mat w;
  //! The H matrix for initialization.
  arma::mat h;
  //! Whether initial W is given.
  bool wIsGiven;
  //! Whether initial H is given.
  bool hIsGiven;
};

} // namespace amf
} // namespace mlpack

#endif
