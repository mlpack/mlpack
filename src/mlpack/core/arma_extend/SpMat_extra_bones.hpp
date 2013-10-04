/**
 * @file SpMat_extra_bones.hpp
 * @author Ryan Curtin
 *
 * Add a batch constructor for SpMat, if the version is older than 3.810.0.
 */
#if ARMA_VERSION_MAJOR == 3 && ARMA_VERSION_MINOR < 810
template<typename T1, typename T2> inline SpMat(
    const Base<uword, T1>& locations,
    const Base<eT, T2>& values,
    const bool sort_locations = true);

template<typename T1, typename T2> inline SpMat(
    const Base<uword, T1>& locations,
    const Base<eT, T2>& values,
    const uword n_rows,
    const uword n_cols,
    const bool sort_locations = true);
#endif
