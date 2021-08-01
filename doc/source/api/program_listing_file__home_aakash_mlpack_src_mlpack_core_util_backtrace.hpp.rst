
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_backtrace.hpp:

Program Listing for File backtrace.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_backtrace.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/backtrace.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef __MLPACK_CORE_UTIL_BACKTRACE_HPP
   #define __MLPACK_CORE_UTIL_BACKTRACE_HPP
   
   #include <string>
   #include <vector>
   
   namespace mlpack {
   
   class Backtrace
   {
    public:
   #ifdef HAS_BFD_DL
   
     Backtrace(int maxDepth = 32);
   #else
   
     Backtrace();
   #endif
     std::string ToString();
   
    private:
     static void GetAddress(int maxDepth);
   
     static void DecodeAddress(long address);
   
     static void DemangleFunction();
   
     struct Frames
     {
       void *address;
       const char* function;
       const char* file;
       unsigned line;
     } static frame;
   
     static std::vector<Frames> stack;
   };
   
   }; // namespace mlpack
   
   #endif
