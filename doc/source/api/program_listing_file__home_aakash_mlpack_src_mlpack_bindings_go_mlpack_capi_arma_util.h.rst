
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_arma_util.h:

Program Listing for File arma_util.h
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_arma_util.h>` (``/home/aakash/mlpack/src/mlpack/bindings/go/mlpack/capi/arma_util.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_MLPACK_ARMAUTIL_H
   #define MLPACK_BINDINGS_GO_MLPACK_ARMAUTIL_H
   
   #include <stdint.h>
   #include <stddef.h>
   #include <stdbool.h>
   
   #if defined(__cplusplus) || defined(c_plusplus)
   extern "C" {
   #endif
   
   void mlpackToArmaMat(const char* identifier,
                        double* mat,
                        const size_t row,
                        const size_t col);
   
   void mlpackToArmaUmat(const char* identifier,
                         double* mat,
                         const size_t row,
                         const size_t col);
   
   void mlpackToArmaRow(const char* identifier,
                        double* rowvec,
                        const size_t elem);
   
   void mlpackToArmaUrow(const char* identifier,
                         double* rowvec,
                         const size_t elem);
   
   void mlpackToArmaCol(const char* identifier,
                        double* colvec,
                        const size_t elem);
   
   void mlpackToArmaUcol(const char* identifier,
                         double* colvec,
                         const size_t elem);
   
   void* mlpackArmaPtrMat(const char* identifier);
   
   void* mlpackArmaPtrUmat(const char* identifier);
   
   void* mlpackArmaPtrRow(const char* identifier);
   
   void* mlpackArmaPtrUrow(const char* identifier);
   
   void* mlpackArmaPtrCol(const char* identifier);
   
   void* mlpackArmaPtrUcol(const char* identifier);
   
   int mlpackNumRowMat(const char* identifier);
   
   int mlpackNumColMat(const char* identifier);
   
   int mlpackNumElemMat(const char* identifier);
   
   int mlpackNumRowUmat(const char* identifier);
   
   int mlpackNumColUmat(const char* identifier);
   
   int mlpackNumElemUmat(const char* identifier);
   
   int mlpackNumElemRow(const char* identifier);
   
   int mlpackNumElemUrow(const char* identifier);
   
   int mlpackNumElemCol(const char* identifier);
   
   int mlpackNumElemUcol(const char* identifier);
   
   void mlpackToArmaMatWithInfo(const char* identifier,
                                const bool* dimensions,
                                double* memptr,
                                const size_t rows,
                                const size_t cols);
   
   int mlpackArmaMatWithInfoElements(const char* identifier);
   
   int mlpackArmaMatWithInfoRows(const char* identifier);
   
   int mlpackArmaMatWithInfoCols(const char* identifier);
   
   void* mlpackArmaPtrMatWithInfoPtr(const char* identifier);
   
   #if defined(__cplusplus) || defined(c_plusplus)
   }
   #endif
   
   #endif
