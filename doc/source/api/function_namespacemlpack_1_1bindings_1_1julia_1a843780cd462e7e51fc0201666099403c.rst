.. _exhale_function_namespacemlpack_1_1bindings_1_1julia_1a843780cd462e7e51fc0201666099403c:

Function mlpack::bindings::julia::GetJuliaType< std::string >
=============================================================

- Defined in :ref:`file__home_aakash_mlpack_src_mlpack_bindings_julia_get_julia_type.hpp`


Function Documentation
----------------------


.. doxygenfunction:: mlpack::bindings::julia::GetJuliaType< std::string >(util::ParamData&, const typename std::enable_if<!util::IsStdVector<std::string>::value>::type *, const typename std::enable_if<!arma::is_arma_type<std::string>::value>::type *, const typename std::enable_if<!std::is_same<std::string, std::tuple<data::DatasetInfo, arma::mat>>::value>::type *, const typename std::enable_if<!data::HasSerialize<std::string>::value>::type *)
