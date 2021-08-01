
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_hpt_hpt.hpp:

Program Listing for File hpt.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_hpt_hpt.hpp>` (``/home/aakash/mlpack/src/mlpack/core/hpt/hpt.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_HPT_HPT_HPP
   #define MLPACK_CORE_HPT_HPT_HPP
   
   #include <mlpack/core/cv/meta_info_extractor.hpp>
   #include <mlpack/core/hpt/deduce_hp_types.hpp>
   #include <ensmallen.hpp>
   
   namespace mlpack {
   namespace hpt {
   
   template<typename MLAlgorithm,
            typename Metric,
            template<typename, typename, typename, typename, typename> class CV,
            typename OptimizerType = ens::GridSearch,
            typename MatType = arma::mat,
            typename PredictionsType =
                typename cv::MetaInfoExtractor<MLAlgorithm,
                    MatType>::PredictionsType,
            typename WeightsType =
                typename cv::MetaInfoExtractor<MLAlgorithm, MatType,
                    PredictionsType>::WeightsType>
   class HyperParameterTuner
   {
    public:
     template<typename... CVArgs>
     HyperParameterTuner(const CVArgs& ...args);
   
     OptimizerType& Optimizer() { return optimizer; }
   
     double RelativeDelta() const { return relativeDelta; }
   
     double& RelativeDelta() { return relativeDelta; }
   
     double MinDelta() const { return minDelta; }
   
     double& MinDelta() { return minDelta; }
   
     template<typename... Args>
     TupleOfHyperParameters<Args...> Optimize(const Args&... args);
   
     double BestObjective() const { return bestObjective; }
   
     const MLAlgorithm& BestModel() const { return bestModel; }
   
     MLAlgorithm& BestModel() { return bestModel; }
   
    private:
     template<typename OriginalMetric>
     struct Negated
     {
       static double Evaluate(MLAlgorithm& model,
                              const MatType& xs,
                              const PredictionsType& ys)
       { return -OriginalMetric::Evaluate(model, xs, ys); }
     };
   
     using CVType = typename std::conditional<Metric::NeedsMinimization,
         CV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
         CV<MLAlgorithm, Negated<Metric>, MatType, PredictionsType,
             WeightsType>>::type;
   
   
     CVType cv;
   
     OptimizerType optimizer;
   
     double bestObjective;
   
     MLAlgorithm bestModel;
   
     double relativeDelta;
   
     double minDelta;
   
     template<typename Tuple, size_t I>
     using IsPreFixed = IsPreFixedArg<typename std::tuple_element<I, Tuple>::type>;
   
     template<typename Tuple, size_t I>
     using IsArithmetic = std::is_arithmetic<typename std::remove_reference<
         typename std::tuple_element<I, Tuple>::type>::type>;
   
     template<size_t I /* Index of the next argument to handle. */,
              typename ArgsTuple,
              typename... FixedArgs,
              typename = std::enable_if_t<I == std::tuple_size<ArgsTuple>::value>>
     inline void InitAndOptimize(
         const ArgsTuple& args,
         arma::mat& bestParams,
         data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
         FixedArgs... fixedArgs);
   
     template<size_t I /* Index of the next argument to handle. */,
              typename ArgsTuple,
              typename... FixedArgs,
              typename = std::enable_if_t<(I < std::tuple_size<ArgsTuple>::value)>,
              typename = std::enable_if_t<IsPreFixed<ArgsTuple, I>::value>>
     inline void InitAndOptimize(
         const ArgsTuple& args,
         arma::mat& bestParams,
         data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
         FixedArgs... fixedArgs);
   
     template<size_t I /* Index of the next argument to handle. */,
              typename ArgsTuple,
              typename... FixedArgs,
              typename = std::enable_if_t<(I < std::tuple_size<ArgsTuple>::value)>,
              typename = std::enable_if_t<!IsPreFixed<ArgsTuple, I>::value &&
                      IsArithmetic<ArgsTuple, I>::value>,
              typename = void>
     inline void InitAndOptimize(
         const ArgsTuple& args,
         arma::mat& bestParams,
         data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
         FixedArgs... fixedArgs);
   
     template<size_t I /* Index of the next argument to handle. */,
              typename ArgsTuple,
              typename... FixedArgs,
              typename = std::enable_if_t<(I < std::tuple_size<ArgsTuple>::value)>,
              typename = std::enable_if_t<!IsPreFixed<ArgsTuple, I>::value &&
                      !IsArithmetic<ArgsTuple, I>::value>,
              typename = void,
              typename = void>
     inline void InitAndOptimize(
         const ArgsTuple& args,
         arma::mat& bestParams,
         data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
         FixedArgs... fixedArgs);
   
     template<typename TupleType,
              size_t I /* Index of the element in vector to handle. */,
              typename... Args,
              typename = typename
                  std::enable_if_t<(I < std::tuple_size<TupleType>::value)>>
     inline TupleType VectorToTuple(const arma::vec& vector, const Args&... args);
   
     template<typename TupleType,
              size_t I /* Index of the element in vector to handle. */,
              typename... Args,
              typename = typename
                  std::enable_if_t<I == std::tuple_size<TupleType>::value>,
              typename = void>
     inline TupleType VectorToTuple(const arma::vec& vector, const Args&... args);
   };
   
   } // namespace hpt
   } // namespace mlpack
   
   // Include implementation
   #include "hpt_impl.hpp"
   
   #endif
