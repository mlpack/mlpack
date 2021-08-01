
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_arma_util.cpp:

Program Listing for File arma_util.cpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_arma_util.cpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/mlpack/capi/arma_util.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/bindings/go/mlpack/capi/arma_util.h>
   #include "arma_util.hpp"
   #include "io_util.hpp"
   #include <mlpack/core/util/io.hpp>
   
   namespace mlpack {
   namespace util {
   
   extern "C" {
   
   void mlpackToArmaMat(const char* identifier, double* mat,
                        const size_t row, const size_t col)
   {
     // Advanced constructor.
     arma::mat m(mat, row, col, false, true);
   
     // Set input parameter with corresponding matrix in IO.
     SetParam(identifier, m);
   }
   
   void mlpackToArmaUmat(const char* identifier, double* mat,
                         const size_t row, const size_t col)
   {
     // Advanced constructor.
     arma::mat m(mat, row, col, false, true);
   
     arma::Mat<size_t> matr = arma::conv_to<arma::Mat<size_t>>::from(m);
   
     // Set input parameter with corresponding matrix in IO.
     SetParam(identifier, matr);
   }
   
   void mlpackToArmaRow(const char* identifier, double* rowvec, const size_t elem)
   {
     // Advanced constructor.
     arma::rowvec m(rowvec, elem, false, true);
   
     // Set input parameter with corresponding row in IO.
     SetParam(identifier, m);
   }
   
   void mlpackToArmaUrow(const char* identifier, double* rowvec, const size_t elem)
   {
     // Advanced constructor.
     arma::rowvec m(rowvec, elem, false, true);
   
     arma::Row<size_t> matr = arma::conv_to<arma::Row<size_t>>::from(m);
   
     // Set input parameter with corresponding row in IO.
     SetParam(identifier, matr);
   }
   
   void mlpackToArmaCol(const char* identifier, double* colvec, const size_t elem)
   {
     // Advanced constructor.
     arma::colvec m(colvec, elem, false, true);
   
     // Set input parameter with corresponding column in IO.
     SetParam(identifier, m);
   }
   
   void mlpackToArmaUcol(const char* identifier, double* colvec, const size_t elem)
   {
     // Advanced constructor.
     arma::colvec m(colvec, elem, false, true);
   
     arma::Col<size_t> matr = arma::conv_to<arma::Col<size_t>>::from(m);
   
     // Set input parameter with corresponding column in IO.
     SetParam(identifier, matr);
   }
   void* mlpackArmaPtrMat(const char* identifier)
   {
     arma::mat& output = IO::GetParam<arma::mat>(identifier);
     if (output.is_empty())
     {
       return NULL;
     }
     void* ptr = GetMemory(output);
     return ptr;
   }
   
   void* mlpackArmaPtrUmat(const char* identifier)
   {
     arma::Mat<size_t>& m = IO::GetParam<arma::Mat<size_t>>(identifier);
   
     arma::mat output = arma::conv_to<arma::mat>::from(m);
     if (output.is_empty())
     {
       return NULL;
     }
     void* ptr = GetMemory(output);
     return ptr;
   }
   
   void* mlpackArmaPtrRow(const char* identifier)
   {
     arma::Row<double>& output = IO::GetParam<arma::Row<double>>(identifier);
     if (output.is_empty())
     {
       return NULL;
     }
     void* ptr = GetMemory(output);
     return ptr;
   }
   
   void* mlpackArmaPtrUrow(const char* identifier)
   {
     arma::Row<size_t>& m = IO::GetParam<arma::Row<size_t>>(identifier);
   
     arma::Row<double> output = arma::conv_to<arma::Row<double>>::from(m);
     if (output.is_empty())
     {
       return NULL;
     }
     void* ptr = GetMemory(output);
     return ptr;
   }
   
   void* mlpackArmaPtrCol(const char* identifier)
   {
     arma::Col<double>& output = IO::GetParam<arma::Col<double>>(identifier);
     if (output.is_empty())
     {
       return NULL;
     }
     void* ptr = GetMemory(output);
     return ptr;
   }
   
   void* mlpackArmaPtrUcol(const char* identifier)
   {
     arma::Col<size_t>& m = IO::GetParam<arma::Col<size_t>>(identifier);
   
     arma::Col<double> output = arma::conv_to<arma::Col<double>>::from(m);
     if (output.is_empty())
     {
       return NULL;
     }
     void* ptr = GetMemory(output);
     return ptr;
   }
   
   int mlpackNumRowMat(const char* identifier)
   {
     return IO::GetParam<arma::mat>(identifier).n_rows;
   }
   
   int mlpackNumColMat(const char* identifier)
   {
     return IO::GetParam<arma::mat>(identifier).n_cols;
   }
   
   int mlpackNumElemMat(const char* identifier)
   {
     return IO::GetParam<arma::mat>(identifier).n_elem;
   }
   
   int mlpackNumRowUmat(const char* identifier)
   {
     return IO::GetParam<arma::Mat<size_t>>(identifier).n_rows;
   }
   
   int mlpackNumColUmat(const char* identifier)
   {
     return IO::GetParam<arma::Mat<size_t>>(identifier).n_cols;
   }
   
   int mlpackNumElemUmat(const char* identifier)
   {
     return IO::GetParam<arma::Mat<size_t>>(identifier).n_elem;
   }
   
   int mlpackNumElemRow(const char* identifier)
   {
     return IO::GetParam<arma::Row<double>>(identifier).n_elem;
   }
   
   int mlpackNumElemUrow(const char* identifier)
   {
     return IO::GetParam<arma::Row<size_t>>(identifier).n_elem;
   }
   
   int mlpackNumElemCol(const char* identifier)
   {
     return IO::GetParam<arma::Col<double>>(identifier).n_elem;
   }
   
   int mlpackNumElemUcol(const char* identifier)
   {
     return IO::GetParam<arma::Col<size_t>>(identifier).n_elem;
   }
   
   void mlpackToArmaMatWithInfo(const char* identifier,
                                const bool* dimensions,
                                double* memptr,
                                const size_t rows,
                                const size_t cols)
   {
     data::DatasetInfo d(rows);
     for (size_t i = 0; i < d.Dimensionality(); ++i)
     {
       d.Type(i) = (dimensions[i]) ? data::Datatype::categorical :
           data::Datatype::numeric;
     }
   
     arma::mat m(memptr, rows, cols, false, true);
     std::get<0>(IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
         identifier)) = std::move(d);
     std::get<1>(IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
         identifier)) = std::move(m);
     IO::SetPassed(identifier);
   }
   
   int mlpackArmaMatWithInfoElements(const char* identifier)
   {
     typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
     return std::get<1>(IO::GetParam<TupleType>(identifier)).n_elem;
   }
   
   int mlpackArmaMatWithInfoRows(const char* identifier)
   {
     typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
     return std::get<1>(IO::GetParam<TupleType>(identifier)).n_rows;
   }
   
   int mlpackArmaMatWithInfoCols(const char* identifier)
   {
     typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
     return std::get<1>(IO::GetParam<TupleType>(identifier)).n_cols;
   }
   
   void* mlpackArmaPtrMatWithInfoPtr(const char* identifier)
   {
     typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
     arma::mat& m = std::get<1>(IO::GetParam<TupleType>(identifier));
     if (m.is_empty())
     {
       return NULL;
     }
     void* ptr = GetMemory(m);
     return ptr;
   }
   
   } // extern "C"
   
   } // namespace util
   } // namespace mlpack
   
