
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_log.hpp:

Program Listing for File log.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_log.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/log.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_LOG_HPP
   #define MLPACK_CORE_UTIL_LOG_HPP
   
   #include <string>
   #include <mlpack/mlpack_export.hpp>
   
   #include "prefixedoutstream.hpp"
   #include "nulloutstream.hpp"
   
   namespace mlpack {
   
   class Log
   {
    public:
     static void Assert(bool condition,
                        const std::string& message = "Assert Failed.");
   
     // We only use PrefixedOutStream if the program is compiled with debug
     // symbols.
   #ifdef DEBUG
     static MLPACK_EXPORT util::PrefixedOutStream Debug;
   #else
     static MLPACK_EXPORT util::NullOutStream Debug;
   #endif
   
     static MLPACK_EXPORT util::PrefixedOutStream Info;
   
     static MLPACK_EXPORT util::PrefixedOutStream Warn;
   
     static MLPACK_EXPORT util::PrefixedOutStream Fatal;
   
     static std::ostream& cout;
   };
   
   }; // namespace mlpack
   
   #endif
