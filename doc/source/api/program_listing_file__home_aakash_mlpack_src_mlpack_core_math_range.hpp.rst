
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_range.hpp:

Program Listing for File range.hpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_range.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/range.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_MATH_RANGE_HPP
   #define MLPACK_CORE_MATH_RANGE_HPP
   
   namespace mlpack {
   namespace math {
   
   template<typename T>
   class RangeType;
   
   typedef RangeType<double> Range;
   
   template<typename T = double>
   class RangeType
   {
    private:
     T lo; 
     T hi; 
   
    public:
     inline RangeType();
   
     /***
      * Initialize a range to enclose only the given point (lo = point, hi =
      * point).
      *
      * @param point Point that this range will enclose.
      */
     inline RangeType(const T point);
   
     inline RangeType(const T lo, const T hi);
   
     inline T Lo() const { return lo; }
     inline T& Lo() { return lo; }
   
     inline T Hi() const { return hi; }
     inline T& Hi() { return hi; }
   
     inline T Width() const;
   
     inline T Mid() const;
   
     inline RangeType& operator|=(const RangeType& rhs);
   
     inline RangeType operator|(const RangeType& rhs) const;
   
     inline RangeType& operator&=(const RangeType& rhs);
   
     inline RangeType operator&(const RangeType& rhs) const;
   
     inline RangeType& operator*=(const T d);
   
     inline RangeType operator*(const T d) const;
   
     template<typename TT>
     friend inline RangeType<TT> operator*(const TT d, const RangeType<TT>& r);
   
     inline bool operator==(const RangeType& rhs) const;
   
     inline bool operator!=(const RangeType& rhs) const;
   
     inline bool operator<(const RangeType& rhs) const;
   
     inline bool operator>(const RangeType& rhs) const;
   
     inline bool Contains(const T d) const;
   
     inline bool Contains(const RangeType& r) const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   };
   
   } // namespace math
   } // namespace mlpack
   
   // Include inlined implementation.
   #include "range_impl.hpp"
   
   #endif // MLPACK_CORE_MATH_RANGE_HPP
