/**
 * @file core/util/distr_param.hpp
 * @author Conrad Saderson
 * @author Ryan Curton
 *
 * A simple mlpack's `DistrParam` replacement for implementations
 * from`coot::distr_param` or `arma::distr_param`. Since both are identical
 * implementations we can copy it directly from Armadillo and use it in both
 * cases internally.
 *
 * This will allow to use the same DistrParam for coot and arma matrices
 * without any changes.
 *
 * As the implementation is inspired heavily from Armadillo it is necessary to
 * add two different licenses: one for Armadillo and another for mlpack.
 *
 * https://gitlab.com/conradsnicta/armadillo-code/-/blob/10.8.x/include/armadillo_bits/distr_param.hpp
 *
 * Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
 * Copyright 2008-2016 National ICT Australia (NICTA)
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ------------------------------------------------------------------------
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_DISTR_PARAM_HPP
#define MLPACK_CORE_UTIL_DISTR_PARAM_HPP

namespace mlpack {

class DistrParam
{
 public:
  inline DistrParam() :
      state(0),
      a_int(0),
      b_int(0),
      a_double(0),
      b_double(0)
  {
    // Nothing to do here.
  }

  inline explicit DistrParam(const int a, const int b) :
      state(1),
      a_int(a),
      b_int(b),
      a_double(double(a)),
      b_double(double(b))
  {
    // Nothing to do here.
  }

  inline explicit DistrParam(const double a, const double b) :
      state(2),
      a_int(int(a)),
      b_int(int(b)),
      a_double(a),
      b_double(b)
  {
      // Nothing to do here.
  }

  operator arma::distr_param()
  {
    if (state == 0)
      return arma::distr_param();
    else if (state == 1)
      return arma::distr_param(a_int, b_int);
    else
      return arma::distr_param(a_double, b_double);
  }

#ifdef MLPACK_HAS_COOT

  operator coot::distr_param()
  {
    if (state == 0)
      return coot::distr_param();
    else if (state == 1)
      return coot::distr_param(a_int, b_int);
    else
      return coot::distr_param(a_double, b_double);
  }

#endif

  size_t state;

 private:
  int a_int;
  int b_int;

  double a_double;
  double b_double;
};

} // namespace mlpack

#endif
