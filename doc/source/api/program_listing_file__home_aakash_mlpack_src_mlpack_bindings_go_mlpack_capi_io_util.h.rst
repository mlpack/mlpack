
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_io_util.h:

Program Listing for File io_util.h
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_io_util.h>` (``/home/aakash/mlpack/src/mlpack/bindings/go/mlpack/capi/io_util.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_MLPACK_IO_UTIL_H
   #define MLPACK_BINDINGS_GO_MLPACK_IO_UTIL_H
   
   #include <stdint.h>
   #include <stddef.h>
   #include <stdbool.h>
   
   #if defined(__cplusplus) || defined(c_plusplus)
   extern "C" {
   #endif
   
   void mlpackSetParamDouble(const char* identifier, double value);
   
   void mlpackSetParamInt(const char* identifier, int value);
   
   void mlpackSetParamFloat(const char* identifier, float value);
   
   void mlpackSetParamBool(const char* identifier, bool value);
   
   void mlpackSetParamString(const char* identifier, const char* value);
   
   void mlpackSetParamPtr(const char* identifier, const double* ptr);
   
   void mlpackSetParamVectorInt(const char* identifier,
                                const long long* ints,
                                const size_t length);
   
   void mlpackSetParamVectorStr(const char* identifier,
                                const char* str,
                                const size_t element);
   
   void mlpackSetParamVectorStrLen(const char* identifier,
                                   const size_t length);
   
   bool mlpackHasParam(const char* identifier);
   
   const char* mlpackGetParamString(const char* identifier);
   
   double mlpackGetParamDouble(const char* identifier);
   
   int mlpackGetParamInt(const char* identifier);
   
   bool mlpackGetParamBool(const char* identifier);
   
   void* mlpackGetVecIntPtr(const char* identifier);
   
   const char* mlpackGetVecStringPtr(const char* identifier, const size_t i);
   
   int mlpackVecIntSize(const char* identifier);
   
   int mlpackVecStringSize(const char* identifier);
   
   void mlpackSetPassed(const char* name);
   
   void mlpackResetTimers();
   
   void mlpackEnableTimers();
   
   void mlpackDisableBacktrace();
   
   void mlpackEnableVerbose();
   
   void mlpackDisableVerbose();
   
   void mlpackClearSettings();
   
   void mlpackRestoreSettings(const char* name);
   
   #if defined(__cplusplus) || defined(c_plusplus)
   }
   #endif
   
   #endif
