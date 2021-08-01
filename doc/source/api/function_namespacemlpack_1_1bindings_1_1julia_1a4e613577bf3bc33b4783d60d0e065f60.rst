.. _exhale_function_namespacemlpack_1_1bindings_1_1julia_1a4e613577bf3bc33b4783d60d0e065f60:

Function mlpack::bindings::julia::GetJuliaType< int >
=====================================================

- Defined in :ref:`file__home_aakash_mlpack_src_mlpack_bindings_julia_get_julia_type.hpp`


Function Documentation
----------------------


.. doxygenfunction:: mlpack::bindings::julia::GetJuliaType< int >(util::ParamData&, const typename std::enable_if<!util::IsStdVector<int>::value>::type *, const typename std::enable_if<!arma::is_arma_type<int>::value>::type *, const typename std::enable_if<!std::is_same<int, std::tuple<data::DatasetInfo, arma::mat>>::value>::type *, const typename std::enable_if<!data::HasSerialize<int>::value>::type *)
