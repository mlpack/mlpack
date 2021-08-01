
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_dists_laplace_distribution.hpp:

Program Listing for File laplace_distribution.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_dists_laplace_distribution.hpp>` (``/home/aakash/mlpack/src/mlpack/core/dists/laplace_distribution.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   /*
    * @file core/dists/laplace_distribution.hpp
    * @author Zhihao Lou
    * @author Rohan Raj
    *
    * Laplace (double exponential) distribution used in SA.
    *
    * mlpack is free software; you may redistribute it and/or modify it under the
    * terms of the 3-clause BSD license.  You should have received a copy of the
    * 3-clause BSD license along with mlpack.  If not, see
    * http://www.opensource.org/licenses/BSD-3-Clause for more information.
    */
   
   #ifndef MLPACK_CORE_DISTRIBUTIONS_LAPLACE_DISTRIBUTION_HPP
   #define MLPACK_CORE_DISTRIBUTIONS_LAPLACE_DISTRIBUTION_HPP
   
   namespace mlpack {
   namespace distribution {
   
   class LaplaceDistribution
   {
    public:
     LaplaceDistribution() : scale(0) { }
   
     LaplaceDistribution(const size_t dimensionality, const double scale) :
         mean(arma::zeros<arma::vec>(dimensionality)), scale(scale) { }
   
     LaplaceDistribution(const arma::vec& mean, const double scale) :
         mean(mean), scale(scale) { }
   
     size_t Dimensionality() const { return mean.n_elem; }
   
     double Probability(const arma::vec& observation) const
     {
       return exp(LogProbability(observation));
     }
   
     void Probability(const arma::mat& x, arma::vec& probabilities) const;
   
     double LogProbability(const arma::vec& observation) const;
   
     void LogProbability(const arma::mat& x, arma::vec& logProbabilities) const
     {
       logProbabilities.set_size(x.n_cols);
       for (size_t i = 0; i < x.n_cols; ++i)
       {
         logProbabilities(i) = LogProbability(x.unsafe_col(i));
       }
     }
   
     arma::vec Random() const
     {
       arma::vec result(mean.n_elem);
       result.randu();
   
       // Convert from uniform distribution to Laplace distribution.
       // arma::sign() does not exist in Armadillo < 3.920 so we have to do this
       // elementwise.
       for (size_t i = 0; i < result.n_elem; ++i)
       {
         if (result[i] < 0.5)
           result[i] = mean[i] + scale * std::log(1 + 2.0 * (result[i] - 0.5));
         else
           result[i] = mean[i] - scale * std::log(1 - 2.0 * (result[i] - 0.5));
       }
   
       return result;
     }
   
     void Estimate(const arma::mat& observations);
   
     void Estimate(const arma::mat& observations,
                   const arma::vec& probabilities);
   
     const arma::vec& Mean() const { return mean; }
     arma::vec& Mean() { return mean; }
   
     double Scale() const { return scale; }
     double& Scale() { return scale; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(mean));
       ar(CEREAL_NVP(scale));
     }
   
    private:
     arma::vec mean;
     double scale;
   };
   
   } // namespace distribution
   } // namespace mlpack
   
   #endif
