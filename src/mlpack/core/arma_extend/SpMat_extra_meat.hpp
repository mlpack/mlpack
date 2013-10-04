/**
 * @file SpMat_extra_meat.hpp
 * @author Ryan Curtin
 *
 * Take the Armadillo batch sparse matrix constructor function from newer
 * Armadillo versions and port it to versions earlier than 3.810.0.
 */
#if ARMA_VERSION_MAJOR == 3 && ARMA_VERSION_MINOR < 810

//! Insert a large number of values at once.
//! locations.row[0] should be row indices, locations.row[1] should be column indices,
//! and values should be the corresponding values.
//! If sort_locations is false, then it is assumed that the locations and values
//! are already sorted in column-major ordering.
template<typename eT>
template<typename T1, typename T2>
inline
SpMat<eT>::SpMat(const Base<uword,T1>& locations_expr, const Base<eT,T2>& vals_expr, const bool sort_locations)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL)
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);

  const unwrap<T1>         locs_tmp( locations_expr.get_ref() );
  const Mat<uword>& locs = locs_tmp.M;

  const unwrap<T2> vals_tmp( vals_expr.get_ref() );
  const Mat<eT>& vals = vals_tmp.M;

  arma_debug_check( (vals.is_vec() == false), "SpMat::SpMat(): given 'values' object is not a vector" );

  arma_debug_check((locs.n_cols != vals.n_elem), "SpMat::SpMat(): number of locations is different than number of values");

  // If there are no elements in the list, max() will fail.
  if (locs.n_cols == 0)
    {
    init(0, 0);
    return;
    }

  arma_debug_check((locs.n_rows != 2), "SpMat::SpMat(): locations matrix must have two rows");

  // Automatically determine size (and check if it's sorted).
  uvec bounds = arma::max(locs, 1);
  init(bounds[0] + 1, bounds[1] + 1);

  // Resize to correct number of elements.
  mem_resize(vals.n_elem);

  // Reset column pointers to zero.
  arrayops::inplace_set(access::rwp(col_ptrs), uword(0), n_cols + 1);

  bool actually_sorted = true;
  if(sort_locations == true)
    {
    // sort_index() uses std::sort() which may use quicksort... so we better
    // make sure it's not already sorted before taking an O(N^2) sort penalty.
    for (uword i = 1; i < locs.n_cols; ++i)
      {
      if ((locs.at(1, i) < locs.at(1, i - 1)) || (locs.at(1, i) == locs.at(1, i - 1) && locs.at(0, i) <= locs.at(0, i - 1)))
        {
        actually_sorted = false;
        break;
        }
      }

    if(actually_sorted == false)
      {
      // This may not be the fastest possible implementation but it maximizes code reuse.
      Col<uword> abslocs(locs.n_cols);

      for (uword i = 0; i < locs.n_cols; ++i)
        {
        abslocs[i] = locs.at(1, i) * n_rows + locs.at(0, i);
        }

      // Now we will sort with sort_index().
      uvec sorted_indices = sort_index(abslocs); // Ascending sort.

      // Now we add the elements in this sorted order.
      for (uword i = 0; i < sorted_indices.n_elem; ++i)
        {
        arma_debug_check((locs.at(0, sorted_indices[i]) >= n_rows), "SpMat::SpMat(): invalid row index");
        arma_debug_check((locs.at(1, sorted_indices[i]) >= n_cols), "SpMat::SpMat(): invalid column index");

        access::rw(values[i])      = vals[sorted_indices[i]];
        access::rw(row_indices[i]) = locs.at(0, sorted_indices[i]);

        access::rw(col_ptrs[locs.at(1, sorted_indices[i]) + 1])++;
        }
      }
    }
  if( (sort_locations == false) || (actually_sorted == true) )
    {
    // Now set the values and row indices correctly.
    // Increment the column pointers in each column (so they are column "counts").
    for (uword i = 0; i < vals.n_elem; ++i)
      {
      arma_debug_check((locs.at(0, i) >= n_rows), "SpMat::SpMat(): invalid row index");
      arma_debug_check((locs.at(1, i) >= n_cols), "SpMat::SpMat(): invalid column index");

      // Check ordering in debug mode.
      if(i > 0)
        {
        arma_debug_check
          (
          ( (locs.at(1, i) < locs.at(1, i - 1)) || (locs.at(1, i) == locs.at(1, i - 1) && locs.at(0, i) < locs.at(0, i - 1)) ),
          "SpMat::SpMat(): out of order points; either pass sort_locations = true, or sort points in column-major ordering"
          );
        arma_debug_check((locs.at(1, i) == locs.at(1, i - 1) && locs.at(0, i) == locs.at(0, i - 1)), "SpMat::SpMat(): two identical point locations in list");
        }

      access::rw(values[i])      = vals[i];
      access::rw(row_indices[i]) = locs.at(0, i);

      access::rw(col_ptrs[locs.at(1, i) + 1])++;
      }
    }

  // Now fix the column pointers.
  for (uword i = 0; i <= n_cols; ++i)
    {
    access::rw(col_ptrs[i + 1]) += col_ptrs[i];
    }
  }



//! Insert a large number of values at once.
//! locations.row[0] should be row indices, locations.row[1] should be column indices,
//! and values should be the corresponding values.
//! If sort_locations is false, then it is assumed that the locations and values
//! are already sorted in column-major ordering.
//! In this constructor the size is explicitly given.
template<typename eT>
template<typename T1, typename T2>
inline
SpMat<eT>::SpMat(const Base<uword,T1>& locations_expr, const Base<eT,T2>& vals_expr, const uword in_n_rows, const uword in_n_cols, const bool sort_locations)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL)
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);

  init(in_n_rows, in_n_cols);

  const unwrap<T1>         locs_tmp( locations_expr.get_ref() );
  const Mat<uword>& locs = locs_tmp.M;

  const unwrap<T2> vals_tmp( vals_expr.get_ref() );
  const Mat<eT>& vals = vals_tmp.M;

  arma_debug_check( (vals.is_vec() == false), "SpMat::SpMat(): given 'values' object is not a vector" );

  arma_debug_check((locs.n_rows != 2), "SpMat::SpMat(): locations matrix must have two rows");

  arma_debug_check((locs.n_cols != vals.n_elem), "SpMat::SpMat(): number of locations is different than number of values");

  // Resize to correct number of elements.
  mem_resize(vals.n_elem);

  // Reset column pointers to zero.
  arrayops::inplace_set(access::rwp(col_ptrs), uword(0), n_cols + 1);

  bool actually_sorted = true;
  if(sort_locations == true)
    {
    // sort_index() uses std::sort() which may use quicksort... so we better
    // make sure it's not already sorted before taking an O(N^2) sort penalty.
    for (uword i = 1; i < locs.n_cols; ++i)
      {
      if ((locs.at(1, i) < locs.at(1, i - 1)) || (locs.at(1, i) == locs.at(1, i - 1) && locs.at(0, i) <= locs.at(0, i - 1)))
        {
        actually_sorted = false;
        break;
        }
      }

    if(actually_sorted == false)
      {
      // This may not be the fastest possible implementation but it maximizes code reuse.
      Col<uword> abslocs(locs.n_cols);

      for (uword i = 0; i < locs.n_cols; ++i)
        {
        abslocs[i] = locs.at(1, i) * n_rows + locs.at(0, i);
        }

      // Now we will sort with sort_index().
      uvec sorted_indices = sort_index(abslocs); // Ascending sort.

      // Now we add the elements in this sorted order.
      for (uword i = 0; i < sorted_indices.n_elem; ++i)
        {
        arma_debug_check((locs.at(0, sorted_indices[i]) >= n_rows), "SpMat::SpMat(): invalid row index");
        arma_debug_check((locs.at(1, sorted_indices[i]) >= n_cols), "SpMat::SpMat(): invalid column index");

        access::rw(values[i])      = vals[sorted_indices[i]];
        access::rw(row_indices[i]) = locs.at(0, sorted_indices[i]);

        access::rw(col_ptrs[locs.at(1, sorted_indices[i]) + 1])++;
        }
      }
    }

  if( (sort_locations == false) || (actually_sorted == true) )
    {
    // Now set the values and row indices correctly.
    // Increment the column pointers in each column (so they are column "counts").
    for (uword i = 0; i < vals.n_elem; ++i)
      {
      arma_debug_check((locs.at(0, i) >= n_rows), "SpMat::SpMat(): invalid row index");
      arma_debug_check((locs.at(1, i) >= n_cols), "SpMat::SpMat(): invalid column index");

      // Check ordering in debug mode.
      if(i > 0)
        {
        arma_debug_check
          (
          ( (locs.at(1, i) < locs.at(1, i - 1)) || (locs.at(1, i) == locs.at(1, i - 1) && locs.at(0, i) < locs.at(0, i - 1)) ),
          "SpMat::SpMat(): out of order points; either pass sort_locations = true or sort points in column-major ordering"
          );
        arma_debug_check((locs.at(1, i) == locs.at(1, i - 1) && locs.at(0, i) == locs.at(0, i - 1)), "SpMat::SpMat(): two identical point locations in list");
        }

      access::rw(values[i])      = vals[i];
      access::rw(row_indices[i]) = locs.at(0, i);

      access::rw(col_ptrs[locs.at(1, i) + 1])++;
      }
    }

  // Now fix the column pointers.
  for (uword i = 0; i <= n_cols; ++i)
    {
    access::rw(col_ptrs[i + 1]) += col_ptrs[i];
    }
  }

#endif
