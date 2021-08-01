
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_normalization_no_normalization.hpp:

Program Listing for File no_normalization.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_normalization_no_normalization.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/normalization/no_normalization.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_NORMALIZATION_NO_NORMALIZATION_HPP
   #define MLPACK_METHODS_CF_NORMALIZATION_NO_NORMALIZATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace cf {
   
   class NoNormalization
   {
    public:
     // Empty constructor.
     NoNormalization() { }
   
     template<typename MatType>
     inline void Normalize(const MatType& /* data */) const { }
   
     inline double Denormalize(const size_t /* user */,
                               const size_t /* item */,
                               const double rating) const
     {
       return rating;
     }
   
     inline void Denormalize(const arma::Mat<size_t>& /* combinations */,
                             const arma::vec& /* predictions */) const
     { }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
