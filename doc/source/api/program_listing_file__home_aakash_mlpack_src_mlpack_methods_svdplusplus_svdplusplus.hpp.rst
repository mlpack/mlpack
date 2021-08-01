
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_svdplusplus_svdplusplus.hpp:

Program Listing for File svdplusplus.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_svdplusplus_svdplusplus.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/svdplusplus/svdplusplus.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SVDPLUSPLUS_SVDPLUSPLUS_HPP
   #define MLPACK_METHODS_SVDPLUSPLUS_SVDPLUSPLUS_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/cf/cf.hpp>
   
   #include <ensmallen.hpp>
   
   #include "svdplusplus_function.hpp"
   
   namespace mlpack {
   namespace svd {
   
   template<typename OptimizerType = ens::StandardSGD>
   class SVDPlusPlus
   {
    public:
     SVDPlusPlus(const size_t iterations = 10,
                 const double alpha = 0.001,
                 const double lambda = 0.1);
   
     void Apply(const arma::mat& data,
                const arma::mat& implicitData,
                const size_t rank,
                arma::mat& u,
                arma::mat& v,
                arma::vec& p,
                arma::vec& q,
                arma::mat& y);
   
     void Apply(const arma::mat& data,
                const size_t rank,
                arma::mat& u,
                arma::mat& v,
                arma::vec& p,
                arma::vec& q,
                arma::mat& y);
   
     static void CleanData(const arma::mat& implicitData,
                           arma::sp_mat& cleanedData,
                           const arma::mat& data);
   
    private:
     size_t iterations;
     double alpha;
     double lambda;
   };
   
   } // namespace svd
   } // namespace mlpack
   
   // Include implementation.
   #include "svdplusplus_impl.hpp"
   
   #endif
