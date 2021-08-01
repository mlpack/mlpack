
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_dists_normal_distribution.hpp:

Program Listing for File normal_distribution.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_dists_normal_distribution.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/dists/normal_distribution.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_HPP
   #define MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "../activation_functions/logistic_function.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <typename DataType = arma::mat>
   class NormalDistribution
   {
    public:
     NormalDistribution();
   
     NormalDistribution(const DataType& mean, const DataType& sigma);
   
     DataType Probability(const DataType& observation) const
     {
       return arma::exp(LogProbability(observation));
     }
   
     DataType LogProbability(const DataType& observation) const;
   
     void ProbBackward(const DataType& observation,
                       DataType& dmu,
                       DataType& dsigma) const;
   
     void Probability(const DataType& x, DataType& probabilities) const
     {
       probabilities = Probability(x);
     }
   
     void LogProbability(const DataType& x, DataType& probabilities) const
     {
       probabilities = LogProbability(x);
     }
   
     DataType Sample() const;
   
     const DataType& Mean() const { return mean; }
   
     DataType& Mean() { return mean; }
   
     const DataType& StandardDeviation() const { return sigma; }
   
     DataType& StandardDeviation() { return sigma; }
   
     size_t Dimensionality() const { return mean.n_elem; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     DataType mean;
   
     DataType sigma;
   }; // class NormalDistribution
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "normal_distribution_impl.hpp"
   
   #endif
