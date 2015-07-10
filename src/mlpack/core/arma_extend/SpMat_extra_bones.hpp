/**
 * @file SpMat_extra_bones.hpp
 * @author Ryan Curtin
 *
 * Add a batch constructor for SpMat, if the version is older than 3.810.0, and
 * also a serialize() function for Armadillo.
 */
template<typename Archive>
void serialize(Archive& ar, const unsigned int version);

#if ARMA_VERSION_MAJOR == 3 && ARMA_VERSION_MINOR < 810
template<typename T1, typename T2>
inline SpMat(
    const Base<uword, T1>& locations,
    const Base<eT, T2>& values,
    const bool sort_locations = true);

template<typename T1, typename T2>
inline SpMat(
    const Base<uword, T1>& locations,
    const Base<eT, T2>& values,
    const uword n_rows,
    const uword n_cols,
    const bool sort_locations = true);
#endif

#if ARMA_VERSION_MAJOR == 3 && ARMA_VERSION_MINOR < 920
template<typename T1, typename T2, typename T3>
inline SpMat(
    const Base<uword, T1>& rowind,
    const Base<uword, T2>& colptr,
    const Base<eT, T3>& values,
    const uword n_rows,
    const uword n_cols);
#endif

/*
 * Extra functions for SpMat<eT>
 * Adding definition of row_col_iterator to generalize with Mat<eT>::row_col_iterator
 */
#if ARMA_VERSION_MAJOR < 4 || \
    (ARMA_VERSION_MAJOR == 4 && ARMA_VERSION_MINOR < 349)
typedef iterator row_col_iterator;
typedef const_iterator const_row_col_iterator;

// begin for iterator row_col_iterator
inline const_row_col_iterator begin_row_col() const;
inline row_col_iterator begin_row_col();

// end for iterator row_col_iterator
inline const_row_col_iterator end_row_col() const;
inline row_col_iterator end_row_col();
#endif
