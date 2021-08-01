
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_io_util.cpp:

Program Listing for File io_util.cpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_io_util.cpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/mlpack/capi/io_util.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/bindings/go/mlpack/capi/io_util.h>
   #include "io_util.hpp"
   #include <mlpack/core/util/io.hpp>
   
   namespace mlpack {
   
   extern "C" {
   
   void mlpackSetParamDouble(const char* identifier, double value)
   {
     util::SetParam(identifier, value);
   }
   
   void mlpackSetParamInt(const char* identifier, int value)
   {
     util::SetParam(identifier, value);
   }
   
   void mlpackSetParamFloat(const char* identifier, float value)
   {
     util::SetParam(identifier, value);
   }
   
   void mlpackSetParamBool(const char* identifier, bool value)
   {
     util::SetParam(identifier, value);
   }
   
   void mlpackSetParamString(const char* identifier, const char* value)
   {
     IO::GetParam<std::string>(identifier) = value;
   }
   
   void mlpackSetParamVectorInt(const char* identifier,
                                const long long* ints,
                                const size_t length)
   {
     // Create a std::vector<int> object; unfortunately this requires copying the
     // vector elements.
     std::vector<int> vec(length);
     for (size_t i = 0; i < length; ++i)
       vec[i] = ints[i];
   
     IO::GetParam<std::vector<int>>(identifier) = std::move(vec);
     IO::SetPassed(identifier);
   }
   
   void mlpackSetParamVectorStrLen(const char* identifier,
                                   const size_t length)
   {
     IO::GetParam<std::vector<std::string>>(identifier).clear();
     IO::GetParam<std::vector<std::string>>(identifier).resize(length);
     IO::SetPassed(identifier);
   }
   
   void mlpackSetParamVectorStr(const char* identifier,
                                const char* str,
                                const size_t element)
   {
     IO::GetParam<std::vector<std::string>>(identifier)[element] =
         std::string(str);
   }
   
   void mlpackSetParamPtr(const char* identifier,
                          const double* ptr)
   {
     util::SetParamPtr(identifier, ptr);
   }
   
   bool mlpackHasParam(const char* identifier)
   {
     return IO::HasParam(identifier);
   }
   
   const char* mlpackGetParamString(const char* identifier)
   {
     return IO::GetParam<std::string>(identifier).c_str();
   }
   
   double mlpackGetParamDouble(const char* identifier)
   {
     return IO::GetParam<double>(identifier);
   }
   
   int mlpackGetParamInt(const char* identifier)
   {
     return IO::GetParam<int>(identifier);
   }
   
   bool mlpackGetParamBool(const char* identifier)
   {
     return IO::GetParam<bool>(identifier);
   }
   
   void* mlpackGetVecIntPtr(const char* identifier)
   {
     const size_t size = mlpackVecIntSize(identifier);
     long long* ints = new long long[size];
   
     for (size_t i = 0; i < size; i++)
       ints[i] = IO::GetParam<std::vector<int>>(identifier)[i];
   
     return ints;
   }
   
   const char* mlpackGetVecStringPtr(const char* identifier, const size_t i)
   {
     return IO::GetParam<std::vector<std::string>>(identifier)[i].c_str();
   }
   
   int mlpackVecIntSize(const char* identifier)
   {
     return IO::GetParam<std::vector<int>>(identifier).size();
   }
   
   int mlpackVecStringSize(const char* identifier)
   {
     return IO::GetParam<std::vector<std::string>>(identifier).size();
   }
   
   void mlpackSetPassed(const char* name)
   {
     IO::SetPassed(name);
   }
   
   void mlpackResetTimers()
   {
     IO::GetSingleton().timer.Reset();
   }
   
   void mlpackEnableTimers()
   {
     Timer::EnableTiming();
   }
   
   void mlpackDisableBacktrace()
   {
     Log::Fatal.backtrace = false;
   }
   
   void mlpackEnableVerbose()
   {
     Log::Info.ignoreInput = false;
   }
   
   void mlpackDisableVerbose()
   {
     Log::Info.ignoreInput = true;
   }
   
   void mlpackClearSettings()
   {
     IO::ClearSettings();
   }
   
   void mlpackRestoreSettings(const char* name)
   {
     IO::RestoreSettings(name);
   }
   
   } // extern C
   
   } // namespace mlpack
