
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cv_k_fold_cv.hpp:

Program Listing for File k_fold_cv.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cv_k_fold_cv.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cv/k_fold_cv.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CV_K_FOLD_CV_HPP
   #define MLPACK_CORE_CV_K_FOLD_CV_HPP
   
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
   class KFoldCV
   {
    public:
     KFoldCV(const size_t k,
             const MatType& xs,
             const PredictionsType& ys,
             const bool shuffle = true);
   
     KFoldCV(const size_t k,
             const MatType& xs,
             const PredictionsType& ys,
             const size_t numClasses,
             const bool shuffle = true);
   
     KFoldCV(const size_t k,
             const MatType& xs,
             const data::DatasetInfo& datasetInfo,
             const PredictionsType& ys,
             const size_t numClasses,
             const bool shuffle = true);
   
     KFoldCV(const size_t k,
             const MatType& xs,
             const PredictionsType& ys,
             const WeightsType& weights,
             const bool shuffle = true);
   
     KFoldCV(const size_t k,
             const MatType& xs,
             const PredictionsType& ys,
             const size_t numClasses,
             const WeightsType& weights,
             const bool shuffle = true);
   
     KFoldCV(const size_t k,
             const MatType& xs,
             const data::DatasetInfo& datasetInfo,
             const PredictionsType& ys,
             const size_t numClasses,
             const WeightsType& weights,
             const bool shuffle = true);
   
     template<typename... MLAlgorithmArgs>
     double Evaluate(const MLAlgorithmArgs& ...args);
   
     MLAlgorithm& Model();
   
    private:
     using Base = CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>;
   
    public:
     template<bool Enabled = !Base::MIE::SupportsWeights,
              typename = typename std::enable_if<Enabled>::type>
     void Shuffle();
   
     template<bool Enabled = Base::MIE::SupportsWeights,
              typename = typename std::enable_if<Enabled>::type,
              typename = void>
     void Shuffle();
   
    private:
     Base base;
   
     const size_t k;
   
     MatType xs;
     PredictionsType ys;
     WeightsType weights;
   
     size_t lastBinSize;
   
     size_t binSize;
   
     std::unique_ptr<MLAlgorithm> modelPtr;
   
     KFoldCV(Base&& base,
             const size_t k,
             const MatType& xs,
             const PredictionsType& ys,
             const bool shuffle);
   
     KFoldCV(Base&& base,
             const size_t k,
             const MatType& xs,
             const PredictionsType& ys,
             const WeightsType& weights,
             const bool shuffle);
   
     template<typename DataType>
     void InitKFoldCVMat(const DataType& source, DataType& destination);
   
     template<typename... MLAlgorithmArgs,
              bool Enabled = !Base::MIE::SupportsWeights,
              typename = typename std::enable_if<Enabled>::type>
     double TrainAndEvaluate(const MLAlgorithmArgs& ...mlAlgorithmArgs);
   
     template<typename... MLAlgorithmArgs,
              bool Enabled = Base::MIE::SupportsWeights,
              typename = typename std::enable_if<Enabled>::type,
              typename = void>
     double TrainAndEvaluate(const MLAlgorithmArgs& ...mlAlgorithmArgs);
   
     inline size_t ValidationSubsetFirstCol(const size_t i);
   
     template<typename ElementType>
     inline arma::Mat<ElementType> GetTrainingSubset(arma::Mat<ElementType>& m,
                                                     const size_t i);
   
     template<typename ElementType>
     inline arma::Row<ElementType> GetTrainingSubset(arma::Row<ElementType>& r,
                                                     const size_t i);
   
     template<typename ElementType>
     inline arma::Mat<ElementType> GetValidationSubset(arma::Mat<ElementType>& m,
                                                       const size_t i);
   
     template<typename ElementType>
     inline arma::Row<ElementType> GetValidationSubset(arma::Row<ElementType>& r,
                                                       const size_t i);
   };
   
   } // namespace cv
   } // namespace mlpack
   
   // Include implementation
   #include "k_fold_cv_impl.hpp"
   
   #endif
