
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_dists_discrete_distribution.hpp:

Program Listing for File discrete_distribution.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_dists_discrete_distribution.hpp>` (``/home/aakash/mlpack/src/mlpack/core/dists/discrete_distribution.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_HPP
   #define MLPACK_CORE_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/log.hpp>
   #include <mlpack/core/math/random.hpp>
   
   namespace mlpack {
   namespace distribution  {
   
   class DiscreteDistribution
   {
    public:
     DiscreteDistribution() :
         probabilities(std::vector<arma::vec>(1)){ /* Nothing to do. */ }
   
     DiscreteDistribution(const size_t numObservations) :
         probabilities(std::vector<arma::vec>(1,
             arma::ones<arma::vec>(numObservations) / numObservations))
     { /* Nothing to do. */ }
   
     DiscreteDistribution(const arma::Col<size_t>& numObservations)
     {
       for (size_t i = 0; i < numObservations.n_elem; ++i)
       {
         const size_t numObs = size_t(numObservations[i]);
         if (numObs <= 0)
         {
           std::ostringstream oss;
           oss << "number of observations for dimension " << i << " is 0, but "
               << "must be greater than 0";
           throw std::invalid_argument(oss.str());
         }
         probabilities.push_back(arma::ones<arma::vec>(numObs) / numObs);
       }
     }
   
     DiscreteDistribution(const std::vector<arma::vec>& probabilities)
     {
       for (size_t i = 0; i < probabilities.size(); ++i)
       {
         arma::vec temp = probabilities[i];
         double sum = accu(temp);
         if (sum > 0)
           this->probabilities.push_back(temp / sum);
         else
         {
           this->probabilities.push_back(arma::ones<arma::vec>(temp.n_elem)
               / temp.n_elem);
         }
       }
     }
   
     size_t Dimensionality() const { return probabilities.size(); }
   
     double Probability(const arma::vec& observation) const
     {
       double probability = 1.0;
       // Ensure the observation has the same dimension with the probabilities.
       if (observation.n_elem != probabilities.size())
       {
         Log::Fatal << "DiscreteDistribution::Probability(): observation has "
             << "incorrect dimension " << observation.n_elem << " but should have"
             << " dimension " << probabilities.size() << "!" << std::endl;
       }
   
       for (size_t dimension = 0; dimension < observation.n_elem; dimension++)
       {
         // Adding 0.5 helps ensure that we cast the floating point to a size_t
         // correctly.
         const size_t obs = size_t(observation(dimension) + 0.5);
   
         // Ensure that the observation is within the bounds.
         if (obs >= probabilities[dimension].n_elem)
         {
           Log::Fatal << "DiscreteDistribution::Probability(): received "
               << "observation " << obs << "; observation must be in [0, "
               << probabilities[dimension].n_elem << "] for this distribution."
               << std::endl;
         }
         probability *= probabilities[dimension][obs];
       }
   
       return probability;
     }
   
     double LogProbability(const arma::vec& observation) const
     {
       // TODO: consider storing log probabilities instead?
       return log(Probability(observation));
     }
   
     void Probability(const arma::mat& x, arma::vec& probabilities) const
     {
       probabilities.set_size(x.n_cols);
       for (size_t i = 0; i < x.n_cols; ++i)
         probabilities(i) = Probability(x.unsafe_col(i));
     }
   
     void LogProbability(const arma::mat& x, arma::vec& logProbabilities) const
     {
       logProbabilities.set_size(x.n_cols);
       for (size_t i = 0; i < x.n_cols; ++i)
         logProbabilities(i) = log(Probability(x.unsafe_col(i)));
     }
   
     arma::vec Random() const;
   
     void Train(const arma::mat& observations);
   
     void Train(const arma::mat& observations,
                const arma::vec& probabilities);
   
     arma::vec& Probabilities(const size_t dim = 0) { return probabilities[dim]; }
     const arma::vec& Probabilities(const size_t dim = 0) const
     { return probabilities[dim]; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(probabilities));
     }
   
    private:
     std::vector<arma::vec> probabilities;
   };
   
   } // namespace distribution
   } // namespace mlpack
   
   #endif
