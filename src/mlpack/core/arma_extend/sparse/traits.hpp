#if (ARMA_VERSION_MAJOR) >= 3
// Cheap hack so that we can do things with SpMats in Armadillo 3.0.0.

template<typename eT>
struct is_Mat< SpMat<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_Mat< SpCol<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_Mat< SpRow<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_Mat< const SpMat<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_Mat< const SpCol<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_Mat< const SpRow<eT> >
  { static const bool value = true; };



template<typename eT>
struct is_Row< SpRow<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_Row< const SpRow<eT> >
  { static const bool value = true; };



template<typename eT>
struct is_Col< SpCol<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_Col< const SpCol<eT> >
  { static const bool value = true; };



template<typename eT>
struct is_subview< SpSubview<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_subview< const SpSubview<eT> >
  { static const bool value = true; };

#endif
