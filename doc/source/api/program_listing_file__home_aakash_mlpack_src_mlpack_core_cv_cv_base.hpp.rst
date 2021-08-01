
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cv_cv_base.hpp:

Program Listing for File cv_base.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cv_cv_base.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cv/cv_base.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CV_CV_BASE_HPP
   #define MLPACK_CORE_CV_CV_BASE_HPP
   
   #include <mlpack/core/cv/meta_info_extractor.hpp>
   
   namespace mlpack {
   namespace cv {
   
   template<typename MLAlgorithm,
            typename MatType,
            typename PredictionsType,
            typename WeightsType>
   class CVBase
   {
    public:
     using MIE =
         MetaInfoExtractor<MLAlgorithm, MatType, PredictionsType, WeightsType>;
   
     CVBase();
   
     CVBase(const size_t numClasses);
   
     CVBase(const data::DatasetInfo& datasetInfo,
            const size_t numClasses);
   
     static void AssertDataConsistency(const MatType& xs,
                                       const PredictionsType& ys);
   
     static void AssertWeightsConsistency(const MatType& xs,
                                          const WeightsType& weights);
   
     template<typename... MLAlgorithmArgs>
     MLAlgorithm Train(const MatType& xs,
                       const PredictionsType& ys,
                       const MLAlgorithmArgs&... args);
   
     template<typename... MLAlgorithmArgs>
     MLAlgorithm Train(const MatType& xs,
                       const PredictionsType& ys,
                       const WeightsType& weights,
                       const MLAlgorithmArgs&... args);
   
    private:
     static_assert(MIE::IsSupported,
         "The given MLAlgorithm is not supported by MetaInfoExtractor");
   
     const data::DatasetInfo datasetInfo;
     const bool isDatasetInfoPassed;
     size_t numClasses;
   
     static void AssertSizeEquality(const MatType& xs,
                                    const PredictionsType& ys);
   
     static void AssertWeightsSize(const MatType& xs,
                                   const WeightsType& weights);
   
     template<typename... MLAlgorithmArgs,
              bool Enabled = !MIE::TakesNumClasses,
              typename = typename std::enable_if<Enabled>::type>
     MLAlgorithm TrainModel(const MatType& xs,
                            const PredictionsType& ys,
                            const MLAlgorithmArgs&... args);
   
     template<typename... MLAlgorithmArgs,
              bool Enabled = MIE::TakesNumClasses & !MIE::TakesDatasetInfo,
              typename = typename std::enable_if<Enabled>::type,
              typename = void>
     MLAlgorithm TrainModel(const MatType& xs,
                            const PredictionsType& ys,
                            const MLAlgorithmArgs&... args);
   
     template<typename... MLAlgorithmArgs,
              bool Enabled = MIE::TakesNumClasses & MIE::TakesDatasetInfo,
              typename = typename std::enable_if<Enabled>::type,
              typename = void,
              typename = void>
     MLAlgorithm TrainModel(const MatType& xs,
                            const PredictionsType& ys,
                            const MLAlgorithmArgs&... args);
   
     template<typename... MLAlgorithmArgs,
              bool Enabled = !MIE::TakesNumClasses,
              typename = typename std::enable_if<Enabled>::type>
     MLAlgorithm TrainModel(const MatType& xs,
                            const PredictionsType& ys,
                            const WeightsType& weights,
                            const MLAlgorithmArgs&... args);
   
     template<typename... MLAlgorithmArgs,
              bool Enabled = MIE::TakesNumClasses & !MIE::TakesDatasetInfo,
              typename = typename std::enable_if<Enabled>::type,
              typename = void>
     MLAlgorithm TrainModel(const MatType& xs,
                            const PredictionsType& ys,
                            const WeightsType& weights,
                            const MLAlgorithmArgs&... args);
   
     template<typename... MLAlgorithmArgs,
              bool Enabled = MIE::TakesNumClasses & MIE::TakesDatasetInfo,
              typename = typename std::enable_if<Enabled>::type,
              typename = void,
              typename = void>
     MLAlgorithm TrainModel(const MatType& xs,
                            const PredictionsType& ys,
                            const WeightsType& weights,
                            const MLAlgorithmArgs&... args);
   
     template<bool ConstructableWithoutDatasetInfo,
              typename... MLAlgorithmArgs,
              typename =
                  typename std::enable_if<ConstructableWithoutDatasetInfo>::type>
     MLAlgorithm TrainModel(const MatType& xs,
                            const PredictionsType& ys,
                            const MLAlgorithmArgs&... args);
   
     template<bool ConstructableWithoutDatasetInfo,
              typename... MLAlgorithmArgs,
              typename =
                  typename std::enable_if<!ConstructableWithoutDatasetInfo>::type,
              typename = void>
     MLAlgorithm TrainModel(const MatType& xs,
                            const PredictionsType& ys,
                            const MLAlgorithmArgs&... args);
   };
   
   } // namespace cv
   } // namespace mlpack
   
   // Include implementation
   #include "cv_base_impl.hpp"
   
   #endif
