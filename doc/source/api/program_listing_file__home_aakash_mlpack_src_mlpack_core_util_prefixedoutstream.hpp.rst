
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_prefixedoutstream.hpp:

Program Listing for File prefixedoutstream.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_prefixedoutstream.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/prefixedoutstream.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_PREFIXEDOUTSTREAM_HPP
   #define MLPACK_CORE_UTIL_PREFIXEDOUTSTREAM_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace util {
   
   class PrefixedOutStream
   {
    public:
     PrefixedOutStream(std::ostream& destination,
                       const char* prefix,
                       bool ignoreInput = false,
                       bool fatal = false,
                       bool backtrace = true) :
         destination(destination),
         ignoreInput(ignoreInput),
         backtrace(backtrace),
         prefix(prefix),
         // We want the first call to operator<< to prefix the prefix so we set
         // carriageReturned to true.
         carriageReturned(true),
         fatal(fatal)
       { /* nothing to do */ }
   
     PrefixedOutStream& operator<<(bool val);
     PrefixedOutStream& operator<<(short val);
     PrefixedOutStream& operator<<(unsigned short val);
     PrefixedOutStream& operator<<(int val);
     PrefixedOutStream& operator<<(unsigned int val);
     PrefixedOutStream& operator<<(long val);
     PrefixedOutStream& operator<<(unsigned long val);
     PrefixedOutStream& operator<<(float val);
     PrefixedOutStream& operator<<(double val);
     PrefixedOutStream& operator<<(long double val);
     PrefixedOutStream& operator<<(void* val);
     PrefixedOutStream& operator<<(const char* str);
     PrefixedOutStream& operator<<(std::string& str);
     PrefixedOutStream& operator<<(std::streambuf* sb);
     PrefixedOutStream& operator<<(std::ostream& (*pf)(std::ostream&));
     PrefixedOutStream& operator<<(std::ios& (*pf)(std::ios&));
     PrefixedOutStream& operator<<(std::ios_base& (*pf)(std::ios_base&));
   
     template<typename T>
     PrefixedOutStream& operator<<(const T& s);
   
     std::ostream& destination;
   
     bool ignoreInput;
   
     bool backtrace;
   
    private:
     template<typename T>
     typename std::enable_if<!arma::is_arma_type<T>::value>::type
     BaseLogic(const T& val);
   
     template<typename T>
     typename std::enable_if<arma::is_arma_type<T>::value>::type
     BaseLogic(const T& val);
   
     inline void PrefixIfNeeded();
   
     std::string prefix;
   
     bool carriageReturned;
   
     bool fatal;
   };
   
   } // namespace util
   } // namespace mlpack
   
   // Template definitions.
   #include "prefixedoutstream_impl.hpp"
   
   #endif
