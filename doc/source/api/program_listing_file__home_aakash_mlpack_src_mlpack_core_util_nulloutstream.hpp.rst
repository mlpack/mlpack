
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_nulloutstream.hpp:

Program Listing for File nulloutstream.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_nulloutstream.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/nulloutstream.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_NULLOUTSTREAM_HPP
   #define MLPACK_CORE_UTIL_NULLOUTSTREAM_HPP
   
   #include <iostream>
   #include <streambuf>
   #include <string>
   
   namespace mlpack {
   namespace util {
   
   class NullOutStream
   {
    public:
     NullOutStream() { }
   
     NullOutStream(const NullOutStream& /* other */) { }
   
     NullOutStream& operator<<(bool) { return *this; }
     NullOutStream& operator<<(short) { return *this; }
     NullOutStream& operator<<(unsigned short) { return *this; }
     NullOutStream& operator<<(int) { return *this; }
     NullOutStream& operator<<(unsigned int) { return *this; }
     NullOutStream& operator<<(long) { return *this; }
     NullOutStream& operator<<(unsigned long) { return *this; }
     NullOutStream& operator<<(float) { return *this; }
     NullOutStream& operator<<(double) { return *this; }
     NullOutStream& operator<<(long double) { return *this; }
     NullOutStream& operator<<(void*) { return *this; }
     NullOutStream& operator<<(const char*) { return *this; }
     NullOutStream& operator<<(std::string&) { return *this; }
     NullOutStream& operator<<(std::streambuf*) { return *this; }
     NullOutStream& operator<<(std::ostream& (*) (std::ostream&)) { return *this; }
     NullOutStream& operator<<(std::ios& (*) (std::ios&)) { return *this; }
     NullOutStream& operator<<(std::ios_base& (*) (std::ios_base&))
     { return *this; }
   
     template<typename T>
     NullOutStream& operator<<(const T&) { return *this; }
   };
   
   } // namespace util
   } // namespace mlpack
   
   #endif
