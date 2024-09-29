/**
 * @file core/data/load_vec_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templatized load() function defined in load.hpp for
 * vectors.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_VEC_IMPL_HPP
#define MLPACK_CORE_DATA_LOAD_VEC_IMPL_HPP

// In case it hasn't already been included.
#include "load.hpp"

namespace mlpack {
namespace data {

// Load column vector.
template<typename eT>
bool Load(const std::string& filename,
          arma::Col<eT>& vec,
          const bool fatal)
{
  // First load into auxiliary matrix.
  arma::Mat<eT> tmp;
  bool success = Load(filename, tmp, fatal, false);
  if (!success)
  {
    vec.clear();
    return false;
  }

  // Now check the size to see that it is a vector, and return a vector.
  if (tmp.n_cols > 1)
  {
    if (tmp.n_rows > 1)
    {
      // Problem: invalid size!
      if (fatal)
      {
        Log::Fatal << "Matrix in file '" << filename << "' is not a vector, but"
            << " instead has size " << tmp.n_rows << "x" << tmp.n_cols << "!"
            << std::endl;
      }
      else
      {
        Log::Warn << "Matrix in file '" << filename << "' is not a vector, but "
            << "instead has size " << tmp.n_rows << "x" << tmp.n_cols << "!"
            << std::endl;
      }

      vec.clear();
      return false;
    }
    else
    {
      /**
       * It's loaded as a row vector (more than one column).  So we need to
       * manually modify the shape of the matrix.  We can do this without
       * damaging the data since it is only a vector.
       */
      arma::access::rw(tmp.n_rows) = tmp.n_cols;
      arma::access::rw(tmp.n_cols) = 1;

      /**
       * Now we can call the move operator, but it has to be the move operator
       * for Mat, not for Col.  This will avoid copying the data.
       */
      *((arma::Mat<eT>*) &vec) = std::move(tmp);
      return true;
    }
  }
  else
  {
    // It's loaded as a column vector.  We can call the move constructor
    // directly.
    *((arma::Mat<eT>*) &vec) = std::move(tmp);
    return true;
  }
}

// Load row vector.
template<typename eT>
bool Load(const std::string& filename,
          arma::Row<eT>& rowvec,
          const bool fatal)
{
  arma::Mat<eT> tmp;
  bool success = Load(filename, tmp, fatal, false);
  if (!success)
  {
    rowvec.clear();
    return false;
  }

  if (tmp.n_rows > 1)
  {
    if (tmp.n_cols > 1)
    {
      // Problem: invalid size!
      if (fatal)
      {
        Log::Fatal << "Matrix in file '" << filename << "' is not a vector, but"
            << " instead has size " << tmp.n_rows << "x" << tmp.n_cols << "!"
            << std::endl;
      }
      else
      {
        Log::Warn << "Matrix in file '" << filename << "' is not a vector, but "
            << "instead has size " << tmp.n_rows << "x" << tmp.n_cols << "!"
            << std::endl;
      }

      rowvec.clear();
      return false;
    }
    else
    {
      /**
       * It's loaded as a column vector (more than one row).  So we need to
       * manually modify the shape of the matrix.  We can do this without
       * damaging the data since it is only a vector.
       */
      arma::access::rw(tmp.n_cols) = tmp.n_rows;
      arma::access::rw(tmp.n_rows) = 1;

      /**
       * Now we can call the move operator, but it has to be the move operator
       * for Mat, not for Col.  This will avoid copying the data.
       */
      *((arma::Mat<eT>*) &rowvec) = std::move(tmp);
      return true;
    }
  }
  else
  {
    // It's loaded as a row vector.  We can call the move constructor directly.
    *((arma::Mat<eT>*) &rowvec) = std::move(tmp);
    return true;
  }
}

} // namespace data
} // namespace mlpack

#endif
