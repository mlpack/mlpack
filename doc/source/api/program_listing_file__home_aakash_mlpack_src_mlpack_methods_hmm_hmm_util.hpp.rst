
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm_util.hpp:

Program Listing for File hmm_util.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm_util.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hmm/hmm_util.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HMM_HMM_UTIL_HPP
   #define MLPACK_METHODS_HMM_HMM_UTIL_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace hmm {
   
   enum HMMType : char
   {
     DiscreteHMM = 0,
     GaussianHMM,
     GaussianMixtureModelHMM,
     DiagonalGaussianMixtureModelHMM
   };
   
   template<typename ActionType, typename ExtraInfoType = void>
   void LoadHMMAndPerformAction(const std::string& modelFile,
                                ExtraInfoType* x = NULL);
   
   template<typename HMMType>
   void SaveHMM(HMMType& hmm, const std::string& modelFile);
   
   } // namespace hmm
   } // namespace mlpack
   
   #include "hmm_util_impl.hpp"
   
   #endif
