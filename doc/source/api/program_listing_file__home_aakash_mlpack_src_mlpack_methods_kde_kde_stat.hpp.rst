
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kde_kde_stat.hpp:

Program Listing for File kde_stat.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kde_kde_stat.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kde/kde_stat.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KDE_STAT_HPP
   #define MLPACK_METHODS_KDE_STAT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kde {
   
   class KDEStat
   {
    public:
     KDEStat() :
         mcBeta(0),
         mcAlpha(0),
         accumAlpha(0),
         accumError(0)
     { /* Nothing to do.*/ }
   
     template<typename TreeType>
     KDEStat(TreeType& /* node */) :
         mcBeta(0),
         mcAlpha(0),
         accumAlpha(0),
         accumError(0)
     { /* Nothing to do. */ }
   
     inline double MCBeta() const { return mcBeta; }
   
     inline double& MCBeta() { return mcBeta; }
   
     inline double AccumAlpha() const { return accumAlpha; }
   
     inline double& AccumAlpha() { return accumAlpha; }
   
     inline double AccumError() const { return accumError; }
   
     inline double& AccumError() { return accumError; }
   
     inline double MCAlpha() const { return mcAlpha; }
   
     inline double& MCAlpha() { return mcAlpha; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(mcBeta));
       ar(CEREAL_NVP(mcAlpha));
       ar(CEREAL_NVP(accumAlpha));
       ar(CEREAL_NVP(accumError));
     }
   
    private:
     double mcBeta;
   
     double mcAlpha;
   
     double accumAlpha;
   
     double accumError;
   };
   
   } // namespace kde
   } // namespace mlpack
   
   #endif
