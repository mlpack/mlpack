/**
 * @file core/util/using.hpp
 * @author Omar Shrit
 * @author Ryan Curtin
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

#endif

  namespace fill {

    #ifdef MLPACK_HAS_COOT
    struct fill_none  : public arma::fill::fill_class<arma::fill::fill_none>,
                        public coot::fill::fill_class<coot::fill::fill_none> { };

    struct fill_zeros : public arma::fill::fill_class<arma::fill::fill_zeros>,
                        public coot::fill::fill_class<coot::fill::fill_zeros> { };

    struct fill_ones  : public arma::fill::fill_class<arma::fill::fill_ones>,
                        public coot::fill::fill_class<coot::fill::fill_ones> { };

    struct fill_randu  : public arma::fill::fill_class<arma::fill::fill_randu>,
                        public coot::fill::fill_class<coot::fill::fill_randu> { };


    #else
    struct fill_none  : public arma::fill::fill_class<arma::fill::fill_none> { };
    struct fill_zeros : public arma::fill::fill_class<arma::fill::fill_zeros> { };
    struct fill_ones  : public arma::fill::fill_class<arma::fill::fill_ones> { };
    struct fill_randu  : public arma::fill::fill_class<arma::fill::fill_randu> { };
    #endif

    static constexpr fill_none  none;
    static constexpr fill_zeros zeros;
    static constexpr fill_ones  ones;
    static constexpr fill_randu randu;
  } // namespace mlpack::fill


} // namespace mlpack

#endif
