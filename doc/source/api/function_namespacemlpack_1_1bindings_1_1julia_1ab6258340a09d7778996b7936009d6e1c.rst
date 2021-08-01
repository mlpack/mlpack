.. _exhale_function_namespacemlpack_1_1bindings_1_1julia_1ab6258340a09d7778996b7936009d6e1c:

Template Function mlpack::bindings::julia::GetJuliaType(util::ParamData&, const typename std::enable_if<!util::IsStdVector<T>::value>::type \*, const typename std::enable_if<!arma::is_arma_type<T>::value>::type \*, const typename std::enable_if<!std::is_same<T, std::tuple<data::DatasetInfo, arma::mat>>::value>::type \*, const typename std::enable_if<!data::HasSerialize<T>::value>::type \*)
========================================================================================================================================================================================================================================================================================================================================================================================================

- Defined in :ref:`file__home_aakash_mlpack_src_mlpack_bindings_julia_get_julia_type.hpp`


Function Documentation
----------------------


.. doxygenfunction:: mlpack::bindings::julia::GetJuliaType(util::ParamData&, const typename std::enable_if<!util::IsStdVector<T>::value>::type *, const typename std::enable_if<!arma::is_arma_type<T>::value>::type *, const typename std::enable_if<!std::is_same<T, std::tuple<data::DatasetInfo, arma::mat>>::value>::type *, const typename std::enable_if<!data::HasSerialize<T>::value>::type *)
