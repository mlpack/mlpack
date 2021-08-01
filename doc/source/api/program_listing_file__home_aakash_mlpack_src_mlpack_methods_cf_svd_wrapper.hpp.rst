
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_svd_wrapper.hpp:

Program Listing for File svd_wrapper.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_svd_wrapper.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/svd_wrapper.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SVDWRAPPER_HPP
   #define MLPACK_METHODS_SVDWRAPPER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack
   {
   namespace cf
   {
   
   class DummyClass {};
   
   template<class Factorizer = DummyClass>
   class SVDWrapper
   {
    public:
     // empty constructor
     SVDWrapper(const Factorizer& factorizer = Factorizer()) :
         factorizer(factorizer)
     {
       // Nothing to do here.
     }
   
     double Apply(const arma::mat& V,
                  arma::mat& W,
                  arma::mat& sigma,
                  arma::mat& H) const;
     double Apply(const arma::mat& V,
                  size_t r,
                  arma::mat& W,
                  arma::mat& H) const;
   
    private:
     Factorizer factorizer;
   }; // class SVDWrapper
   
   typedef SVDWrapper<DummyClass> ArmaSVDFactorizer;
   
   } // namespace cf
   } // namespace mlpack
   
   #include "svd_wrapper_impl.hpp"
   
   #endif
