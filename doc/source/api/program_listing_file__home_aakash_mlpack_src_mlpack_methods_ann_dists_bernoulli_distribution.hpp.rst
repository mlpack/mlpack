
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_dists_bernoulli_distribution.hpp:

Program Listing for File bernoulli_distribution.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_dists_bernoulli_distribution.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/dists/bernoulli_distribution.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_DISTRIBUTIONS_BERNOULLI_DISTRIBUTION_HPP
   #define MLPACK_METHODS_ANN_DISTRIBUTIONS_BERNOULLI_DISTRIBUTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "../activation_functions/logistic_function.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <typename DataType = arma::mat>
   class BernoulliDistribution
   {
    public:
     BernoulliDistribution();
   
     BernoulliDistribution(const DataType& param,
                           const bool applyLogistic = true,
                           const double eps = 1e-10);
   
     double Probability(const DataType& observation) const
     {
       return std::exp(LogProbability(observation));
     }
   
     double LogProbability(const DataType& observation) const;
   
     void LogProbBackward(const DataType& observation, DataType& output) const;
   
     DataType Sample() const;
   
     const DataType& Probability() const { return probability; }
   
     DataType& Probability() { return probability; }
   
     const DataType& Logits() const { return logits; }
   
     DataType& Logits() { return logits; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       // We just need to serialize each of the members.
       ar(CEREAL_NVP(probability));
       ar(CEREAL_NVP(logits));
       ar(CEREAL_NVP(applyLogistic));
       ar(CEREAL_NVP(eps));
     }
   
    private:
     DataType probability;
   
     DataType logits;
   
     bool applyLogistic;
   
     double eps;
   }; // class BernoulliDistribution
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "bernoulli_distribution_impl.hpp"
   
   #endif
