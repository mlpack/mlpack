
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_is_naninf.hpp:

Program Listing for File is_naninf.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_is_naninf.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/is_naninf.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_HAS_NANINF_HPP
   #define MLPACK_CORE_DATA_HAS_NANINF_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   
   template<typename T>
   inline bool IsNaNInf(T& val, const std::string& token)
   {
     // See if the token represents a NaN or Inf.
     if ((token.length() == 3) || (token.length() == 4))
     {
       const bool neg = (token[0] == '-');
       const bool pos = (token[0] == '+');
   
       const size_t offset = ((neg || pos) && (token.length() == 4)) ? 1 : 0;
   
       const std::string token2 = token.substr(offset, 3);
   
       if ((token2 == "inf") || (token2 == "Inf") || (token2 == "INF"))
       {
         if (std::numeric_limits<T>::has_infinity)
         {
           val = (!neg) ? std::numeric_limits<T>::infinity() :
               -1 * std::numeric_limits<T>::infinity();
         }
         else
         {
           val = (!neg) ? std::numeric_limits<T>::max() :
               -1 * std::numeric_limits<T>::max();
         }
   
         return true;
       }
       else if ((token2 == "nan") || (token2 == "Nan") || (token2 == "NaN") ||
           (token2 == "NAN") )
       {
         if (std::numeric_limits<T>::has_quiet_NaN)
           val = std::numeric_limits<T>::quiet_NaN();
         else
           val = T(0);
   
         return true;
       }
     }
   
     return false;
   }
   
   } // namespace data
   } // namespace mlpack
   
   #endif
