.. _exhale_function_namespacemlpack_1_1bindings_1_1julia_1a1b3cc6b2b692d2fac7d986d33a08dd34:

Function mlpack::bindings::julia::GetJuliaType< bool >
======================================================

- Defined in :ref:`file__home_aakash_mlpack_src_mlpack_bindings_julia_get_julia_type.hpp`


Function Documentation
----------------------


.. doxygenfunction:: mlpack::bindings::julia::GetJuliaType< bool >(util::ParamData&, const typename std::enable_if<!util::IsStdVector<bool>::value>::type *, const typename std::enable_if<!arma::is_arma_type<bool>::value>::type *, const typename std::enable_if<!std::is_same<bool, std::tuple<data::DatasetInfo, arma::mat>>::value>::type *, const typename std::enable_if<!data::HasSerialize<bool>::value>::type *)
