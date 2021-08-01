
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_julia_julia_util.h:

Program Listing for File julia_util.h
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_julia_julia_util.h>` (``/home/aakash/mlpack/src/mlpack/bindings/julia/julia_util.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_JULIA_JULIA_UTIL_H
   #define MLPACK_BINDINGS_JULIA_JULIA_UTIL_H
   
   #if defined(__cplusplus) || defined(c_plusplus)
   
   #include <cstddef>
   #include <cstdint>
   extern "C"
   {
   #else
   #include <stddef.h>
   #include <stdint.h>
   #endif
   
   void IO_RestoreSettings(const char* programName);
   
   void IO_SetParamInt(const char* paramName, int paramValue);
   
   void IO_SetParamDouble(const char* paramName, double paramValue);
   
   void IO_SetParamString(const char* paramName, const char* paramValue);
   
   void IO_SetParamBool(const char* paramName, bool paramValue);
   
   void IO_SetParamVectorStrLen(const char* paramName,
                                 const size_t length);
   
   void IO_SetParamVectorStrStr(const char* paramName,
                                 const char* str,
                                 const size_t element);
   
   void IO_SetParamVectorInt(const char* paramName,
                              int* ints,
                              const size_t length);
   
   void IO_SetParamMat(const char* paramName,
                        double* memptr,
                        const size_t rows,
                        const size_t cols,
                        const bool pointsAsRows);
   
   void IO_SetParamUMat(const char* paramName,
                         size_t* memptr,
                         const size_t rows,
                         const size_t cols,
                         const bool pointsAsRows);
   
   void IO_SetParamRow(const char* paramName,
                        double* memptr,
                        const size_t cols);
   
   void IO_SetParamURow(const char* paramName,
                         size_t* memptr,
                         const size_t cols);
   
   void IO_SetParamCol(const char* paramName,
                        double* memptr,
                        const size_t rows);
   
   void IO_SetParamUCol(const char* paramName,
                         size_t* memptr,
                         const size_t rows);
   
   void IO_SetParamMatWithInfo(const char* paramName,
                                bool* dimensions,
                                double* memptr,
                                const size_t rows,
                                const size_t cols,
                                const bool pointsAreRows);
   
   int IO_GetParamInt(const char* paramName);
   
   double IO_GetParamDouble(const char* paramName);
   
   const char* IO_GetParamString(const char* paramName);
   
   bool IO_GetParamBool(const char* paramName);
   
   size_t IO_GetParamVectorStrLen(const char* paramName);
   
   const char* IO_GetParamVectorStrStr(const char* paramName, const size_t i);
   
   size_t IO_GetParamVectorIntLen(const char* paramName);
   
   int* IO_GetParamVectorIntPtr(const char* paramName);
   
   size_t IO_GetParamMatRows(const char* paramName);
   
   size_t IO_GetParamMatCols(const char* paramName);
   
   double* IO_GetParamMat(const char* paramName);
   
   size_t IO_GetParamUMatRows(const char* paramName);
   
   size_t IO_GetParamUMatCols(const char* paramName);
   
   size_t* IO_GetParamUMat(const char* paramName);
   
   size_t IO_GetParamColRows(const char* paramName);
   
   double* IO_GetParamCol(const char* paramName);
   
   size_t IO_GetParamUColRows(const char* paramName);
   
   size_t* IO_GetParamUCol(const char* paramName);
   
   size_t IO_GetParamRowCols(const char* paramName);
   
   double* IO_GetParamRow(const char* paramName);
   
   size_t IO_GetParamURowCols(const char* paramName);
   
   size_t* IO_GetParamURow(const char* paramName);
   
   size_t IO_GetParamMatWithInfoRows(const char* paramName);
   
   size_t IO_GetParamMatWithInfoCols(const char* paramName);
   
   bool* IO_GetParamMatWithInfoBoolPtr(const char* paramName);
   
   double* IO_GetParamMatWithInfoPtr(const char* paramName);
   
   void IO_EnableVerbose();
   
   void IO_DisableVerbose();
   
   void IO_ResetTimers();
   
   void IO_SetPassed(const char* paramName);
   
   #if defined(__cplusplus) || defined(c_plusplus)
   }
   #endif
   
   #endif
