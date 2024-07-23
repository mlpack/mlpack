/**
 * @file core/util/using.hpp
 * @author Omar Shrit
 *
 * This is a set of `using` statements to mitigate any possible risks or
 * conflicts with local functions. The compiler is supposed to proritise the
 * following functions to be looked up first. This is to be considered as a
 * replacement to the ADL solution that we had deployed earlier.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_USING_HPP
#define MLPACK_CORE_UTIL_USING_HPP

namespace mlpack {

  /* using for armadillo namespace*/
  using arma::conv_to;
  using arma::exp;
  using arma::distr_param;
  using arma::dot;
  using arma::join_cols;
  using arma::join_rows;
  using arma::log;
  using arma::min;
  using arma::max;
  using arma::mean;
  using arma::norm;
  using arma::normalise;
  using arma::ones;
  using arma::pow;
  using arma::randi;
  using arma::randn;
  using arma::randu;
  using arma::repmat;
  using arma::sign;
  using arma::sqrt;
  using arma::square;
  using arma::sum;
  using arma::trans;
  using arma::vectorise;
  using arma::zeros;

#ifdef MLPACK_HAS_COOT

  /* using for bandicoot namespace*/
  using coot::conv_to;
  using coot::exp;
  using coot::distr_param;
  using coot::dot;
  using coot::join_cols;
  using coot::join_rows;
  using coot::log;
  using coot::min;
  using coot::max;
  using coot::mean;
  using coot::norm;
  using coot::normalise;
  using coot::ones;
  using coot::pow;
  using coot::randi;
  using coot::randn;
  using coot::randu;
  using coot::repmat;
  using coot::sign;
  using coot::sqrt;
  using coot::square;
  using coot::sum;
  using coot::trans;
  using coot::vectorise;
  using coot::zeros;

#endif

} // namespace mlpack

#endif
