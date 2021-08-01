
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cv_simple_cv.hpp:

Program Listing for File simple_cv.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cv_simple_cv.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cv/simple_cv.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CV_SIMPLE_CV_HPP
   #define MLPACK_CORE_CV_SIMPLE_CV_HPP
   
   #include <mlpack/core/cv/meta_info_extractor.hpp>
   #include <mlpack/core/cv/cv_base.hpp>
   
   namespace mlpack {
   namespace cv {
   
   template<typename MLAlgorithm,
            typename Metric,
            typename MatType = arma::mat,
            typename PredictionsType =
                typename MetaInfoExtractor<MLAlgorithm, MatType>::PredictionsType,
            typename WeightsType =
                typename MetaInfoExtractor<MLAlgorithm, MatType,
                    PredictionsType>::WeightsType>
   class SimpleCV
   {
    public:
     template<typename MatInType, typename PredictionsInType>
     SimpleCV(const double validationSize,
              MatInType&& xs,
              PredictionsInType&& ys);
   
     template<typename MatInType, typename PredictionsInType>
     SimpleCV(const double validationSize,
              MatInType&& xs,
              PredictionsInType&& ys,
              const size_t numClasses);
   
     template<typename MatInType, typename PredictionsInType>
     SimpleCV(const double validationSize,
              MatInType&& xs,
              const data::DatasetInfo& datasetInfo,
              PredictionsInType&& ys,
              const size_t numClasses);
   
     template<typename MatInType,
              typename PredictionsInType,
              typename WeightsInType>
     SimpleCV(const double validationSize,
              MatInType&& xs,
              PredictionsInType&& ys,
              WeightsInType&& weights);
   
     template<typename MatInType,
              typename PredictionsInType,
              typename WeightsInType>
     SimpleCV(const double validationSize,
              MatInType&& xs,
              PredictionsInType&& ys,
              const size_t numClasses,
              WeightsInType&& weights);
   
     template<typename MatInType,
              typename PredictionsInType,
              typename WeightsInType>
     SimpleCV(const double validationSize,
              MatInType&& xs,
              const data::DatasetInfo& datasetInfo,
              PredictionsInType&& ys,
              const size_t numClasses,
              WeightsInType&& weights);
   
     template<typename... MLAlgorithmArgs>
     double Evaluate(const MLAlgorithmArgs&... args);
   
     MLAlgorithm& Model();
   
    private:
     using Base = CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>;
   
     Base base;
   
     MatType xs;
     PredictionsType ys;
     WeightsType weights;
   
     MatType trainingXs;
     PredictionsType trainingYs;
     WeightsType trainingWeights;
   
     MatType validationXs;
     PredictionsType validationYs;
   
     std::unique_ptr<MLAlgorithm> modelPtr;
   
     template<typename MatInType,
              typename PredictionsInType>
     SimpleCV(Base&& base,
              const double validationSize,
              MatInType&& xs,
              PredictionsInType&& ys);
   
     template<typename MatInType,
              typename PredictionsInType,
              typename WeightsInType>
     SimpleCV(Base&& base,
              const double validationSize,
              MatInType&& xs,
              PredictionsInType&& ys,
              WeightsInType&& weights);
   
     size_t CalculateAndAssertNumberOfTrainingPoints(const double validationSize);
   
     template<typename ElementType>
     arma::Mat<ElementType> GetSubset(arma::Mat<ElementType>& m,
                                      const size_t firstCol,
                                      const size_t lastCol);
   
     template<typename ElementType>
     arma::Row<ElementType> GetSubset(arma::Row<ElementType>& r,
                                      const size_t firstCol,
                                      const size_t lastCol);
   
     template<typename... MLAlgorithmArgs,
              bool Enabled = !Base::MIE::SupportsWeights,
              typename = typename std::enable_if<Enabled>::type>
     double TrainAndEvaluate(const MLAlgorithmArgs&... args);
   
     template<typename... MLAlgorithmArgs,
              bool Enabled = Base::MIE::SupportsWeights,
              typename = typename std::enable_if<Enabled>::type,
              typename = void>
     double TrainAndEvaluate(const MLAlgorithmArgs&... args);
   };
   
   } // namespace cv
   } // namespace mlpack
   
   // Include implementation
   #include "simple_cv_impl.hpp"
   
   #endif
