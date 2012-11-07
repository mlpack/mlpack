/**
 * @file Row_extra_bones.hpp
 * @author Ryan Curtin
 *
 * Add an extra explicit constructor for sparse vectors, but only if it doesn't
 * already exist (Armadillo 3.4 specific).
 */
#if ARMA_VERSION_MAJOR == 3 && ARMA_VERSION_MINOR == 4
  inline explicit Row(const SpRow<eT>& X);
#endif
