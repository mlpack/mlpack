.. _exhale_function_namespacemlpack_1_1bindings_1_1julia_1a75357eff29c1a0027df10939cb595596:

Function mlpack::bindings::julia::GetJuliaType< double >
========================================================

- Defined in :ref:`file__home_aakash_mlpack_src_mlpack_bindings_julia_get_julia_type.hpp`


Function Documentation
----------------------


.. doxygenfunction:: mlpack::bindings::julia::GetJuliaType< double >(util::ParamData&, const typename std::enable_if<!util::IsStdVector<double>::value>::type *, const typename std::enable_if<!arma::is_arma_type<double>::value>::type *, const typename std::enable_if<!std::is_same<double, std::tuple<data::DatasetInfo, arma::mat>>::value>::type *, const typename std::enable_if<!data::HasSerialize<double>::value>::type *)
