
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_cf.hpp:

Program Listing for File cf.hpp
===============================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_cf.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/cf.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_CF_HPP
   #define MLPACK_METHODS_CF_CF_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/neighbor_search/neighbor_search.hpp>
   #include <mlpack/methods/amf/amf.hpp>
   #include <mlpack/methods/amf/update_rules/nmf_als.hpp>
   #include <mlpack/methods/amf/termination_policies/simple_residue_termination.hpp>
   #include <mlpack/methods/cf/normalization/no_normalization.hpp>
   #include <mlpack/methods/cf/decomposition_policies/nmf_method.hpp>
   #include <mlpack/methods/cf/neighbor_search_policies/lmetric_search.hpp>
   #include <mlpack/methods/cf/interpolation_policies/average_interpolation.hpp>
   #include <set>
   #include <map>
   #include <iostream>
   
   namespace mlpack {
   namespace cf  {
   template<typename DecompositionPolicy = NMFPolicy,
            typename NormalizationType = NoNormalization>
   class CFType
   {
    public:
     CFType(const size_t numUsersForSimilarity = 5, const size_t rank = 0);
   
     template<typename MatType>
     CFType(const MatType& data,
            const DecompositionPolicy& decomposition = DecompositionPolicy(),
            const size_t numUsersForSimilarity = 5,
            const size_t rank = 0,
            const size_t maxIterations = 1000,
            const double minResidue = 1e-5,
            const bool mit = false);
   
     void Train(const arma::mat& data,
                const DecompositionPolicy& decomposition,
                const size_t maxIterations = 1000,
                const double minResidue = 1e-5,
                const bool mit = false);
   
     void Train(const arma::sp_mat& data,
                const DecompositionPolicy& decomposition,
                const size_t maxIterations = 1000,
                const double minResidue = 1e-5,
                const bool mit = false);
   
     void NumUsersForSimilarity(const size_t num)
     {
       if (num < 1)
       {
         Log::Warn << "CFType::NumUsersForSimilarity(): invalid value (< 1) "
             "ignored." << std::endl;
         return;
       }
       this->numUsersForSimilarity = num;
     }
   
     size_t NumUsersForSimilarity() const
     {
       return numUsersForSimilarity;
     }
   
     void Rank(const size_t rankValue)
     {
       this->rank = rankValue;
     }
   
     size_t Rank() const
     {
       return rank;
     }
   
     const DecompositionPolicy& Decomposition() const { return decomposition; }
   
     const arma::sp_mat& CleanedData() const { return cleanedData; }
   
     const NormalizationType& Normalization() const { return normalization; }
   
     template<typename NeighborSearchPolicy = EuclideanSearch,
              typename InterpolationPolicy = AverageInterpolation>
     void GetRecommendations(const size_t numRecs,
                             arma::Mat<size_t>& recommendations);
   
     template<typename NeighborSearchPolicy = EuclideanSearch,
              typename InterpolationPolicy = AverageInterpolation>
     void GetRecommendations(const size_t numRecs,
                             arma::Mat<size_t>& recommendations,
                             const arma::Col<size_t>& users);
   
     static void CleanData(const arma::mat& data, arma::sp_mat& cleanedData);
   
     template<typename NeighborSearchPolicy = EuclideanSearch,
              typename InterpolationPolicy = AverageInterpolation>
     double Predict(const size_t user, const size_t item) const;
   
     template<typename NeighborSearchPolicy = EuclideanSearch,
              typename InterpolationPolicy = AverageInterpolation>
     void Predict(const arma::Mat<size_t>& combinations,
                  arma::vec& predictions) const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t numUsersForSimilarity;
     size_t rank;
     DecompositionPolicy decomposition;
     arma::sp_mat cleanedData;
     NormalizationType normalization;
   
     typedef std::pair<double, size_t> Candidate;
   
     struct CandidateCmp {
       bool operator()(const Candidate& c1, const Candidate& c2)
       {
         return c1.first > c2.first;
       };
     };
   }; // class CFType
   
   } // namespace cf
   } // namespace mlpack
   
   // Include implementation of templated functions.
   #include "cf_impl.hpp"
   
   #endif
