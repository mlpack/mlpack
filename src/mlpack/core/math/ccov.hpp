/**
 * @file ccov.hpp
 * @author Ryan Curtin
 * @author Conrad Sanderson
 *
 * ccov(X) is same as cov(trans(X)) but without the cost of computing trans(X)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_CCOV_HPP
#define MLPACK_CORE_MATH_CCOV_HPP

namespace mlpack {
namespace math /** Miscellaneous math routines. */ {

template<typename eT>
inline
arma::Mat<eT>
ccov(const arma::Mat<eT>& A, const arma::uword norm_type = 0)
{
  if (norm_type > 1)
  {
    Log::Fatal << "ccov(): norm_type must be 0 or 1" << std::endl;
  }
  
  arma::Mat<eT> out;
  
  if (A.is_vec())
  {
    if (A.n_rows == 1)
    {
      out = arma::var(arma::trans(A), norm_type);
    }
    else
    {
      out = arma::var(A, norm_type);
    }
  }
  else
  {
    const arma::uword N = A.n_cols;
    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);
    
    const arma::Col<eT> acc = arma::sum(A, 1);
    
    out = A * arma::trans(A);
    out -= (acc * arma::trans(acc)) / eT(N);
    out /= norm_val;
  }
  
  return out;
}



template<typename T>
inline
arma::Mat< std::complex<T> >
ccov(const arma::Mat< std::complex<T> >& A, const arma::uword norm_type = 0)
{
  if (norm_type > 1)
  {
    Log::Fatal << "ccov(): norm_type must be 0 or 1" << std::endl;
  }
  
  typedef typename std::complex<T> eT;
  
  arma::Mat<eT> out;
  
  if (A.is_vec())
  {
    if (A.n_rows == 1)
    {
      const arma::Mat<T> tmp_mat = arma::var(arma::trans(A), norm_type);
      out.set_size(1,1);
      out[0] = tmp_mat[0];
    }
    else
    {
      const arma::Mat<T> tmp_mat = arma::var(A, norm_type);
      out.set_size(1,1);
      out[0] = tmp_mat[0];
    }
  }
  else
  {
    const arma::uword N = A.n_cols;
    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    const arma::Col<eT> acc = arma::sum(A, 1);

    out = A * arma::trans(arma::conj(A));
    out -= (acc * arma::trans(arma::conj(acc))) / eT(N);
    out /= norm_val;
  }
  
  return out;
}



template<typename eT>
inline
arma::Mat<eT>
ccov(const arma::Mat<eT>& A, const arma::Mat<eT>& B, const arma::uword norm_type = 0)
{
  if (norm_type > 1)
  {
    Log::Fatal << "ccov(): norm_type must be 0 or 1" << std::endl;
  }
  
  arma::Mat<eT> out;

  if (A.is_vec() && B.is_vec())
  {
    if (A.n_elem != B.n_elem)
    {
      Log::Fatal << "ccov(): the number of elements in A and B must match" << std::endl;
    }

    const eT* A_ptr = A.memptr();
    const eT* B_ptr = B.memptr();

    eT A_acc   = eT(0);
    eT B_acc   = eT(0);
    eT out_acc = eT(0);

    const arma::uword N = A.n_elem;

    for (arma::uword i=0; i<N; ++i)
    {
      const eT A_tmp = A_ptr[i];
      const eT B_tmp = B_ptr[i];

      A_acc += A_tmp;
      B_acc += B_tmp;

      out_acc += A_tmp * B_tmp;
    }

    out_acc -= (A_acc * B_acc)/eT(N);

    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    out.set_size(1,1);
    out[0] = out_acc/norm_val;
  }
  else
  {
    if ( (A.n_rows != B.n_rows) || (A.n_cols != B.n_cols) )
    {
      Log::Fatal << "ccov(): size of A and B must match" << std::endl;
    }
    
    const arma::uword N = A.n_cols;
    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    out = A * arma::trans(B);
    out -= (sum(A) * arma::trans(sum(B))) / eT(N);
    out /= norm_val;
  }
  
  return out;
}



template<typename T>
inline
arma::Mat< std::complex<T> >
ccov(const arma::Mat< std::complex<T> >& A, const arma::Mat< std::complex<T> >& B, const arma::uword norm_type = 0)
{
  if (norm_type > 1)
  {
    Log::Fatal << "ccov(): norm_type must be 0 or 1" << std::endl;
  }
  
  typedef typename std::complex<T> eT;
  
  arma::Mat<eT> out;

  if (A.is_vec() && B.is_vec())
  {
    if (A.n_elem != B.n_elem)
    {
      Log::Fatal << "ccov(): the number of elements in A and B must match" << std::endl;
    }

    const eT* A_ptr = A.memptr();
    const eT* B_ptr = B.memptr();

    eT A_acc   = eT(0);
    eT B_acc   = eT(0);
    eT out_acc = eT(0);

    const arma::uword N = A.n_elem;

    for (arma::uword i=0; i<N; ++i)
    {
      const eT A_tmp = A_ptr[i];
      const eT B_tmp = B_ptr[i];

      A_acc += A_tmp;
      B_acc += B_tmp;

      out_acc += std::conj(A_tmp) * B_tmp;
    }

    out_acc -= (std::conj(A_acc) * B_acc)/eT(N);

    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    out.set_size(1,1);
    out[0] = out_acc/norm_val;
  }
  else
  {
    if ( (A.n_rows != B.n_rows) || (A.n_cols != B.n_cols) )
    {
      Log::Fatal << "ccov(): size of A and B must match" << std::endl;
    }

    const arma::uword N = A.n_cols;
    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    out = A * arma::trans(arma::conj(B));
    out -= (sum(A) * arma::trans(arma::conj(arma::sum(B)))) / eT(N);
    out /= norm_val;
  }
  
  return out;
}


} // namespace math
} // namespace mlpack


#endif // MLPACK_CORE_MATH_CCOV_HPP
