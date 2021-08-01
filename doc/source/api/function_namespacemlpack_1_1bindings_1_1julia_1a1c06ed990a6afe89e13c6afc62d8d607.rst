.. _exhale_function_namespacemlpack_1_1bindings_1_1julia_1a1c06ed990a6afe89e13c6afc62d8d607:

Function mlpack::bindings::julia::GetJuliaType< size_t >
========================================================

- Defined in :ref:`file__home_aakash_mlpack_src_mlpack_bindings_julia_get_julia_type.hpp`


Function Documentation
----------------------


.. doxygenfunction:: mlpack::bindings::julia::GetJuliaType< size_t >(util::ParamData&, const typename std::enable_if<!util::IsStdVector<size_t>::value>::type *, const typename std::enable_if<!arma::is_arma_type<size_t>::value>::type *, const typename std::enable_if<!std::is_same<size_t, std::tuple<data::DatasetInfo, arma::mat>>::value>::type *, const typename std::enable_if<!data::HasSerialize<size_t>::value>::type *)
