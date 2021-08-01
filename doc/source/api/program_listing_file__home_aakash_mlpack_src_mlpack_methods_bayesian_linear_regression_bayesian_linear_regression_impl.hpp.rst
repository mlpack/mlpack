
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_bayesian_linear_regression_bayesian_linear_regression_impl.hpp:

Program Listing for File bayesian_linear_regression_impl.hpp
============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_bayesian_linear_regression_bayesian_linear_regression_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/bayesian_linear_regression/bayesian_linear_regression_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_IMPL_HPP
   #define MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_IMPL_HPP
   
   #include "bayesian_linear_regression.hpp"
   
   namespace mlpack {
   namespace regression {
   
   template<typename Archive>
   void BayesianLinearRegression::serialize(Archive& ar,
                                            const uint32_t /* version */)
   {
     ar(CEREAL_NVP(centerData));
     ar(CEREAL_NVP(scaleData));
     ar(CEREAL_NVP(maxIterations));
     ar(CEREAL_NVP(tolerance));
     ar(CEREAL_NVP(dataOffset));
     ar(CEREAL_NVP(dataScale));
     ar(CEREAL_NVP(responsesOffset));
     ar(CEREAL_NVP(alpha));
     ar(CEREAL_NVP(beta));
     ar(CEREAL_NVP(gamma));
     ar(CEREAL_NVP(omega));
     ar(CEREAL_NVP(matCovariance));
   }
   
   } // namespace regression
   } // namespace mlpack
   
   #endif
