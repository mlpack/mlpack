
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_io_util.hpp:

Program Listing for File io_util.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_io_util.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/mlpack/capi/io_util.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_IO_UTIL_HPP
   #define MLPACK_BINDINGS_GO_IO_UTIL_HPP
   
   #include <mlpack/core/util/io.hpp>
   #include <mlpack/core/data/dataset_mapper.hpp>
   
   namespace mlpack {
   namespace util {
   
   template<typename T>
   inline void SetParam(const std::string& identifier, T& value)
   {
     IO::GetParam<T>(identifier) = std::move(value);
   }
   
   template<typename T>
   inline void SetParamPtr(const std::string& identifier,
                           T* value)
   {
     IO::GetParam<T*>(identifier) = value;
   }
   
   template<typename T>
   T* GetParamPtr(const std::string& paramName)
   {
     return IO::GetParam<T*>(paramName);
   }
   
   inline void EnableVerbose()
   {
     Log::Info.ignoreInput = false;
   }
   
   inline void DisableVerbose()
   {
     Log::Info.ignoreInput = true;
   }
   
   inline void DisableBacktrace()
   {
     Log::Fatal.backtrace = false;
   }
   
   inline void ResetTimers()
   {
     // Just get a new object---removes all old timers.
     IO::GetSingleton().timer.Reset();
   }
   
   inline void EnableTimers()
   {
     Timer::EnableTiming();
   }
   
   } // namespace util
   } // namespace mlpack
   
   #endif
