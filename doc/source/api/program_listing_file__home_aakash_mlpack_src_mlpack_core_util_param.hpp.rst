
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_param.hpp:

Program Listing for File param.hpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_param.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_PARAM_HPP
   #define MLPACK_CORE_UTIL_PARAM_HPP
   
   // Required forward declarations.
   namespace mlpack {
   namespace data {
   
   class IncrementPolicy;
   
   template<typename PolicyType, typename InputType>
   class DatasetMapper;
   
   using DatasetInfo = DatasetMapper<IncrementPolicy, std::string>;
   
   } // namespace data
   } // namespace mlpack
   
   // These are ugly, but necessary utility functions we must use to generate a
   // unique identifier inside of the PARAM() module.
   #define JOIN(x, y) JOIN_AGAIN(x, y)
   #define JOIN_AGAIN(x, y) x ## y
   
   #define BINDING_NAME(NAME) static \
       mlpack::util::ProgramName \
       io_programname_dummy_object = mlpack::util::ProgramName(NAME);
   
   #define BINDING_SHORT_DESC(SHORT_DESC) static \
       mlpack::util::ShortDescription \
       io_programshort_desc_dummy_object = mlpack::util::ShortDescription( \
       SHORT_DESC);
   
   #define BINDING_LONG_DESC(LONG_DESC) static \
       mlpack::util::LongDescription \
       io_programlong_desc_dummy_object = mlpack::util::LongDescription( \
       []() { return std::string(LONG_DESC); });
   
   #ifdef __COUNTER__
     #define BINDING_EXAMPLE(EXAMPLE) static \
         mlpack::util::Example \
         JOIN(io_programexample_dummy_object_, __COUNTER__) = \
         mlpack::util::Example( \
         []() { return(std::string(EXAMPLE)); });
   #else
     #define BINDING_EXAMPLE(EXAMPLE) static \
         mlpack::util::Example \
         JOIN(JOIN(io_programexample_dummy_object_, __LINE__), opt) = \
         mlpack::util::Example( \
         []() { return(std::string(EXAMPLE)); });
   #endif
   
   #ifdef __COUNTER__
     #define BINDING_SEE_ALSO(DESCRIPTION, LINK) static \
         mlpack::util::SeeAlso \
         JOIN(io_programsee_also_dummy_object_, __COUNTER__) = \
         mlpack::util::SeeAlso(DESCRIPTION, LINK);
   #else
     #define BINDING_SEE_ALSO(DESCRIPTION, LINK) static \
         mlpack::util::SeeAlso \
         JOIN(JOIN(io_programsee_also_dummy_object_, __LINE__), opt) = \
         mlpack::util::SeeAlso(DESCRIPTION, LINK);
   #endif
   
   #define PARAM_FLAG(ID, DESC, ALIAS) \
       PARAM_IN(bool, ID, DESC, ALIAS, false, false);
   
   #define PARAM_INT_IN(ID, DESC, ALIAS, DEF) \
       PARAM_IN(int, ID, DESC, ALIAS, DEF, false)
   
   #define PARAM_INT_OUT(ID, DESC) \
       PARAM_OUT(int, ID, DESC, "", 0, false)
   
   #define PARAM_DOUBLE_IN(ID, DESC, ALIAS, DEF) \
       PARAM_IN(double, ID, DESC, ALIAS, DEF, false)
   
   #define PARAM_DOUBLE_OUT(ID, DESC) \
       PARAM_OUT(double, ID, DESC, "", 0.0, false)
   
   #define PARAM_STRING_IN(ID, DESC, ALIAS, DEF) \
       PARAM_IN(std::string, ID, DESC, ALIAS, DEF, false)
   
   #define PARAM_STRING_OUT(ID, DESC, ALIAS) \
       PARAM_OUT(std::string, ID, DESC, ALIAS, "", false)
   
   #define PARAM_MATRIX_IN(ID, DESC, ALIAS) \
       PARAM_MATRIX(ID, DESC, ALIAS, false, true, true)
   
   #define PARAM_MATRIX_IN_REQ(ID, DESC, ALIAS) \
       PARAM_MATRIX(ID, DESC, ALIAS, true, true, true)
   
   #define PARAM_MATRIX_OUT(ID, DESC, ALIAS) \
       PARAM_MATRIX(ID, DESC, ALIAS, false, true, false)
   
   #define PARAM_TMATRIX_IN(ID, DESC, ALIAS) \
       PARAM_MATRIX(ID, DESC, ALIAS, false, false, true)
   
   #define PARAM_TMATRIX_IN_REQ(ID, DESC, ALIAS) \
       PARAM_MATRIX(ID, DESC, ALIAS, true, false, true)
   
   #define PARAM_TMATRIX_OUT(ID, DESC, ALIAS) \
       PARAM_MATRIX(ID, DESC, ALIAS, false, false, false)
   
   #define PARAM_UMATRIX_IN(ID, DESC, ALIAS) \
       PARAM_UMATRIX(ID, DESC, ALIAS, false, true, true)
   
   #define PARAM_UMATRIX_IN_REQ(ID, DESC, ALIAS) \
       PARAM_UMATRIX(ID, DESC, ALIAS, true, true, true)
   
   #define PARAM_UMATRIX_OUT(ID, DESC, ALIAS) \
       PARAM_UMATRIX(ID, DESC, ALIAS, false, true, false)
   
   
   #define PARAM_COL_IN(ID, DESC, ALIAS) \
       PARAM_COL(ID, DESC, ALIAS, false, true, true)
   
   #define PARAM_COL_IN_REQ(ID, DESC, ALIAS) \
       PARAM_COL(ID, DESC, ALIAS, true, true, true)
   
   #define PARAM_ROW_IN(ID, DESC, ALIAS) \
       PARAM_ROW(ID, DESC, ALIAS, false, true, true)
   
   #define PARAM_UCOL_IN(ID, DESC, ALIAS) \
       PARAM_UCOL(ID, DESC, ALIAS, false, true, true)
   
   #define PARAM_UROW_IN(ID, DESC, ALIAS) \
       PARAM_UROW(ID, DESC, ALIAS, false, true, true)
   
   #define PARAM_COL_OUT(ID, DESC, ALIAS) \
       PARAM_COL(ID, DESC, ALIAS, false, true, false)
   
   #define PARAM_ROW_OUT(ID, DESC, ALIAS) \
       PARAM_ROW(ID, DESC, ALIAS, false, true, false)
   
   #define PARAM_UCOL_OUT(ID, DESC, ALIAS) \
       PARAM_UCOL(ID, DESC, ALIAS, false, true, false)
   
   #define PARAM_UROW_OUT(ID, DESC, ALIAS) \
       PARAM_UROW(ID, DESC, ALIAS, false, true, false)
   
   #define PARAM_VECTOR_IN(T, ID, DESC, ALIAS) \
       PARAM_IN(std::vector<T>, ID, DESC, ALIAS, std::vector<T>(), false)
   
   #define PARAM_VECTOR_OUT(T, ID, DESC, ALIAS) \
       PARAM_OUT(std::vector<T>, ID, DESC, ALIAS, std::vector<T>(), false)
   
   #define TUPLE_TYPE std::tuple<mlpack::data::DatasetInfo, arma::mat>
   #define PARAM_MATRIX_AND_INFO_IN(ID, DESC, ALIAS) \
       PARAM(TUPLE_TYPE, ID, DESC, ALIAS, \
           "std::tuple<mlpack::data::DatasetInfo, arma::mat>", false, true, true, \
           TUPLE_TYPE())
   
   #define PARAM_MODEL_IN(TYPE, ID, DESC, ALIAS) \
       PARAM_MODEL(TYPE, ID, DESC, ALIAS, false, true)
   
   #define PARAM_MODEL_IN_REQ(TYPE, ID, DESC, ALIAS) \
       PARAM_MODEL(TYPE, ID, DESC, ALIAS, true, true)
   
   #define PARAM_MODEL_OUT(TYPE, ID, DESC, ALIAS) \
       PARAM_MODEL(TYPE, ID, DESC, ALIAS, false, false)
   
   #define PARAM_INT_IN_REQ(ID, DESC, ALIAS) \
       PARAM_IN(int, ID, DESC, ALIAS, 0, true)
   
   #define PARAM_DOUBLE_IN_REQ(ID, DESC, ALIAS) \
       PARAM_IN(double, ID, DESC, ALIAS, 0.0, true)
   
   #define PARAM_STRING_IN_REQ(ID, DESC, ALIAS) \
       PARAM_IN(std::string, ID, DESC, ALIAS, "", true)
   
   #define PARAM_VECTOR_IN_REQ(T, ID, DESC, ALIAS) \
       PARAM_IN(std::vector<T>, ID, DESC, ALIAS, std::vector<T>(), true);
   
   #define PARAM_IN(T, ID, DESC, ALIAS, DEF, REQ) \
       PARAM(T, ID, DESC, ALIAS, #T, REQ, true, false, DEF);
   
   #define PARAM_OUT(T, ID, DESC, ALIAS, DEF, REQ) \
       PARAM(T, ID, DESC, ALIAS, #T, REQ, false, false, DEF);
   
   #define PARAM_MATRIX(ID, DESC, ALIAS, REQ, TRANS, IN) \
       PARAM(arma::mat, ID, DESC, ALIAS, "arma::mat", REQ, IN, \
           TRANS, arma::mat());
   
   #define PARAM_UMATRIX(ID, DESC, ALIAS, REQ, TRANS, IN) \
       PARAM(arma::Mat<size_t>, ID, DESC, ALIAS, "arma::Mat<size_t>", \
           REQ, IN, TRANS, arma::Mat<size_t>());
   
   #define PARAM_COL(ID, DESC, ALIAS, REQ, TRANS, IN) \
       PARAM(arma::vec, ID, DESC, ALIAS, "arma::vec", REQ, IN, TRANS, \
           arma::vec());
   
   #define PARAM_UCOL(ID, DESC, ALIAS, REQ, TRANS, IN) \
       PARAM(arma::Col<size_t>, ID, DESC, ALIAS, "arma::Col<size_t>", \
           REQ, IN, TRANS, arma::Col<size_t>());
   
   #define PARAM_ROW(ID, DESC, ALIAS, REQ, TRANS, IN) \
       PARAM(arma::rowvec, ID, DESC, ALIAS, "arma::rowvec", REQ, IN, \
       TRANS, arma::rowvec());
   
   #define PARAM_UROW(ID, DESC, ALIAS, REQ, TRANS, IN) \
       PARAM(arma::Row<size_t>, ID, DESC, ALIAS, "arma::Row<size_t>", \
       REQ, IN, TRANS, arma::Row<size_t>());
   
   #ifdef __COUNTER__
     #define PARAM(T, ID, DESC, ALIAS, NAME, REQ, IN, TRANS, DEF) \
         static mlpack::util::Option<T> \
         JOIN(io_option_dummy_object_in_, __COUNTER__) \
         (DEF, ID, DESC, ALIAS, NAME, REQ, IN, !TRANS, testName);
   
     // There are no uses of required models, so that is not an option to this
     // macro (it would be easy to add).
     #define PARAM_MODEL(TYPE, ID, DESC, ALIAS, REQ, IN) \
         static mlpack::util::Option<TYPE*> \
         JOIN(io_option_dummy_model_, __COUNTER__) \
         (nullptr, ID, DESC, ALIAS, #TYPE, REQ, IN, false, testName);
   #else
     // We have to do some really bizarre stuff since __COUNTER__ isn't defined. I
     // don't think we can absolutely guarantee success, but it should be "good
     // enough".  We use the __LINE__ macro and the type of the parameter to try
     // and get a good guess at something unique.
     #define PARAM(T, ID, DESC, ALIAS, NAME, REQ, IN, TRANS, DEF) \
         static mlpack::util::Option<T> \
         JOIN(JOIN(io_option_dummy_object_in_, __LINE__), opt) \
         (DEF, ID, DESC, ALIAS, NAME, REQ, IN, !TRANS, testName);
   
     #define PARAM_MODEL(TYPE, ID, DESC, ALIAS, REQ, IN) \
         static mlpack::util::Option<TYPE*> \
         JOIN(JOIN(io_option_dummy_object_model_, __LINE__), opt) \
         (nullptr, ID, DESC, ALIAS, #TYPE, REQ, IN, false, \
         testName);
   #endif
   
   #endif
