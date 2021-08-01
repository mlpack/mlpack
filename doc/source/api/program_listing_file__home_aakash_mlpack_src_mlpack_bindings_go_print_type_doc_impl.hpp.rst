
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_print_type_doc_impl.hpp:

Program Listing for File print_type_doc_impl.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_print_type_doc_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/print_type_doc_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_PRINT_TYPE_DOC_IMPL_HPP
   #define MLPACK_BINDINGS_GO_PRINT_TYPE_DOC_IMPL_HPP
   
   #include "print_type_doc.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace go {
   
   template<typename T>
   std::string PrintTypeDoc(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type*,
       const typename boost::disable_if<util::IsStdVector<T>>::type*,
       const typename boost::disable_if<data::HasSerialize<T>>::type*,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type*)
   {
     // A flag type.
     if (std::is_same<T, bool>::value)
     {
       return "A boolean flag option (`true` or `false`).";
     }
     // An integer.
     else if (std::is_same<T, int>::value)
     {
       return "An integer (i.e., `1`).";
     }
     // A floating point value.
     else if (std::is_same<T, double>::value)
     {
       return "A floating-point number (i.e., `0.5`).";
     }
     // A string.
     else if (std::is_same<T, std::string>::value)
     {
       return "A character string (i.e., `\"hello\"`).";
     }
     // Not sure what it is...
     else
     {
       throw std::invalid_argument("unknown parameter type " + data.cppType);
     }
   }
   
   template<typename T>
   std::string PrintTypeDoc(
       util::ParamData& data,
       const typename std::enable_if<util::IsStdVector<T>::value>::type*)
   {
     if (std::is_same<T, std::vector<int>>::value)
     {
       return "An array of integers; i.e., `[]int{0, 1, 2}`.";
     }
     else if (std::is_same<T, std::vector<std::string>>::value)
     {
       return "An array of strings; i.e., `[]string{\"hello\", \"goodbye\"}`.";
     }
     else
     {
       throw std::invalid_argument("unknown vector type " + data.cppType);
     }
   }
   
   template<typename T>
   std::string PrintTypeDoc(
       util::ParamData& /* data */,
       const typename std::enable_if<arma::is_arma_type<T>::value>::type*)
   {
     if (T::is_col || T::is_row)
     {
       return "A 1-d gonum Matrix (that is, a Matrix where either the number"
              " of rows or number of columns is 1).";
     }
     else
     {
       return "A 2-d gonum Matrix. If the type is not already `float64`, it "
              "will be converted.";
     }
   }
   
   template<typename T>
   std::string PrintTypeDoc(
       util::ParamData& /* data */,
       const typename std::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
   {
     return "A Tuple(matrixWithInfo) containing `float64` data (Data) along with a"
        " boolean array (Categoricals) indicating which dimensions are categorical"
        " (represented by `true`) and which are numeric (represented by `false`)."
        "  The number of elements in the boolean array should be the same as the"
        " dimensionality of the data matrix.  It is expected that each row of the"
        " matrix corresponds to a single data point when calling mlpack bindings.";
   }
   
   template<typename T>
   std::string PrintTypeDoc(
       util::ParamData& /* data */,
       const typename boost::disable_if<arma::is_arma_type<T>>::type*,
       const typename boost::enable_if<data::HasSerialize<T>>::type*)
   {
     return "An mlpack model pointer.  This type holds a pointer to C++ memory "
         "containing the mlpack model.  Note that this means the mlpack model "
         "itself cannot be easily inspected in Go.  However, the pointer can "
         "be passed to subsequent calls to mlpack functions.";
   }
   
   } // namespace go
   } // namespace bindings
   } // namespace mlpack
   
   #endif
