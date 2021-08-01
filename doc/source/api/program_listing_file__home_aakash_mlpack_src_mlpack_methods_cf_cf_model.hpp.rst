
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_cf_model.hpp:

Program Listing for File cf_model.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_cf_model.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/cf_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_CF_MODEL_HPP
   #define MLPACK_METHODS_CF_CF_MODEL_HPP
   
   #include <mlpack/core.hpp>
   #include "cf.hpp"
   
   namespace mlpack {
   namespace cf {
   
   enum NeighborSearchTypes
   {
     COSINE_SEARCH,
     EUCLIDEAN_SEARCH,
     PEARSON_SEARCH
   };
   
   enum InterpolationTypes
   {
     AVERAGE_INTERPOLATION,
     REGRESSION_INTERPOLATION,
     SIMILARITY_INTERPOLATION
   };
   
   class CFWrapperBase
   {
    public:
     CFWrapperBase() { }
   
     virtual CFWrapperBase* Clone() const = 0;
   
     virtual ~CFWrapperBase() { }
   
     virtual void Predict(const NeighborSearchTypes nsType,
                          const InterpolationTypes interpolationType,
                          const arma::Mat<size_t>& combinations,
                          arma::vec& predictions) = 0;
   
     virtual void GetRecommendations(
         const NeighborSearchTypes nsType,
         const InterpolationTypes interpolationType,
         const size_t numRecs,
         arma::Mat<size_t>& recommendations) = 0;
   
     virtual void GetRecommendations(
         const NeighborSearchTypes nsType,
         const InterpolationTypes interpolationType,
         const size_t numRecs,
         arma::Mat<size_t>& recommendations,
         const arma::Col<size_t>& users) = 0;
   };
   
   template<typename DecompositionPolicy, typename NormalizationPolicy>
   class CFWrapper : public CFWrapperBase
   {
    protected:
     typedef CFType<DecompositionPolicy, NormalizationPolicy> CFModelType;
   
    public:
     CFWrapper() { }
   
     CFWrapper(const arma::mat& data,
               const DecompositionPolicy& decomposition,
               const size_t numUsersForSimilarity,
               const size_t rank,
               const size_t maxIterations,
               const size_t minResidue,
               const bool mit) :
         cf(data,
            decomposition,
            numUsersForSimilarity,
            rank,
            maxIterations,
            minResidue,
            mit)
     {
       // Nothing else to do.
     }
   
     virtual CFWrapper* Clone() const { return new CFWrapper(*this); }
   
     virtual ~CFWrapper() { }
   
     CFModelType& CF() { return cf; }
   
     virtual void Predict(const NeighborSearchTypes nsType,
                          const InterpolationTypes interpolationType,
                          const arma::Mat<size_t>& combinations,
                          arma::vec& predictions);
   
     virtual void GetRecommendations(
         const NeighborSearchTypes nsType,
         const InterpolationTypes interpolationType,
         const size_t numRecs,
         arma::Mat<size_t>& recommendations);
   
     virtual void GetRecommendations(
         const NeighborSearchTypes nsType,
         const InterpolationTypes interpolationType,
         const size_t numRecs,
         arma::Mat<size_t>& recommendations,
         const arma::Col<size_t>& users);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(cf));
     }
   
    protected:
     CFModelType cf;
   };
   
   class CFModel
   {
    public:
     enum DecompositionTypes
     {
       NMF,
       BATCH_SVD,
       RANDOMIZED_SVD,
       REG_SVD,
       SVD_COMPLETE,
       SVD_INCOMPLETE,
       BIAS_SVD,
       SVD_PLUS_PLUS
     };
   
     enum NormalizationTypes
     {
       NO_NORMALIZATION,
       ITEM_MEAN_NORMALIZATION,
       USER_MEAN_NORMALIZATION,
       OVERALL_MEAN_NORMALIZATION,
       Z_SCORE_NORMALIZATION
     };
   
    private:
     DecompositionTypes decompositionType;
     NormalizationTypes normalizationType;
   
     CFWrapperBase* cf;
   
    public:
     CFModel();
   
     CFModel(const CFModel& other);
   
     CFModel(CFModel&& other);
   
     CFModel& operator=(const CFModel& other);
   
     CFModel& operator=(CFModel&& other);
   
     ~CFModel();
   
     CFWrapperBase* CF() const { return cf; }
   
     const DecompositionTypes& DecompositionType() const
     {
       return decompositionType;
     }
     DecompositionTypes& DecompositionType()
     {
       return decompositionType;
     }
   
     const NormalizationTypes& NormalizationType() const
     {
       return normalizationType;
     }
     NormalizationTypes& NormalizationType()
     {
       return normalizationType;
     }
   
     void Train(const arma::mat& data,
                const size_t numUsersForSimilarity,
                const size_t rank,
                const size_t maxIterations,
                const double minResidue,
                const bool mit);
   
     void Predict(const NeighborSearchTypes nsType,
                  const InterpolationTypes interpolationType,
                  const arma::Mat<size_t>& combinations,
                  arma::vec& predictions);
   
     void GetRecommendations(const NeighborSearchTypes nsType,
                             const InterpolationTypes interpolationType,
                             const size_t numRecs,
                             arma::Mat<size_t>& recommendations,
                             const arma::Col<size_t>& users);
   
     void GetRecommendations(const NeighborSearchTypes nsType,
                             const InterpolationTypes interpolationType,
                             const size_t numRecs,
                             arma::Mat<size_t>& recommendations);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   };
   
   } // namespace cf
   } // namespace mlpack
   
   // Include implementation.
   #include "cf_model_impl.hpp"
   
   #endif
