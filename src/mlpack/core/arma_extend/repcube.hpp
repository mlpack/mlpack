/**
 * @file repcube.hpp
 * @author Andrew Furey
 *
 * When repcube() is not available (Armadillo < 15.0), provide an internal
 * mlpack implementation that operates the same way.
 */
#ifndef MLPACK_CORE_ARMA_EXTEND_REPCUBE_HPP
#define MLPACK_CORE_ARMA_EXTEND_REPCUBE_HPP

#include <armadillo>

namespace arma {

#if ARMA_VERSION_MAJOR < 15

template<typename eT>
Cube<eT> repcube(const Cube<eT>& X,
                   const uword copies_per_row,
                   const uword copies_per_col,
                   const uword copies_per_slice)
{
  Cube<eT> out;
  out.set_size(X.n_rows * copies_per_row,
               X.n_cols * copies_per_col,
               X.n_slices * copies_per_slice);

  if (out.is_empty())
    return out;

  for (size_t s = 0; s < out.n_slices; s++)
  {
    for (size_t c = 0; c < out.n_cols; c++)
    {
      for (size_t r = 0; r < out.n_rows; r++)
      {
        out.subcube(r, c, s, r, c, s) =
          X.subcube(r / copies_per_row,
                    c / copies_per_col,
                    s / copies_per_slice,
                    r / copies_per_row,
                    c / copies_per_col,
                    s / copies_per_slice);
      }
    }
  }

  return out;
}

template<typename eT>
Cube<eT> repcube(const Mat<eT>& X,
                   const uword copies_per_row,
                   const uword copies_per_col,
                   const uword copies_per_slice)
{
  Cube<eT> out;
  out.set_size(X.n_rows * copies_per_row,
               X.n_cols * copies_per_col,
               copies_per_slice);

  if (out.is_empty())
    return out;

  const size_t X_size = X.n_rows * X.n_cols;
  for (size_t s = 0; s < out.n_slices; s++)
  {
    for (size_t c = 0; c < out.n_cols; c++)
    {
      for (size_t r = 0; r < out.n_rows; r++)
      {
        out.subcube(r, c, s, r, c, s) =
          X.submat(r / copies_per_row, c / copies_per_col,
                   r / copies_per_row, c / copies_per_col);
      }
    }
  }

  return out;
}

#endif

} // namespace mlpack

#endif
