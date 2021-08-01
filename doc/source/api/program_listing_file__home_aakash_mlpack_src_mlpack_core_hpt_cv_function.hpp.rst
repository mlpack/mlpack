
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_hpt_cv_function.hpp:

Program Listing for File cv_function.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_hpt_cv_function.hpp>` (``/home/aakash/mlpack/src/mlpack/core/hpt/cv_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_HPT_CV_FUNCTION_HPP
   #define MLPACK_CORE_HPT_CV_FUNCTION_HPP
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace hpt {
   
   template<typename CVType,
            typename MLAlgorithm,
            size_t TotalArgs,
            typename... BoundArgs>
   class CVFunction
   {
    public:
     CVFunction(CVType& cv,
                data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
                const double relativeDelta,
                const double minDelta,
                const BoundArgs&... args);
   
     double Evaluate(const arma::mat& parameters);
   
     void Gradient(const arma::mat& parameters, arma::mat& gradient);
   
     MLAlgorithm& BestModel() { return bestModel; }
   
    private:
     using BoundArgsTupleType = std::tuple<BoundArgs...>;
   
     static const size_t BoundArgsAmount =
         std::tuple_size<BoundArgsTupleType>::value;
   
     template<size_t BoundArgIndex,
              size_t ParamIndex,
              bool BoundArgsIndexInRange = (BoundArgIndex < BoundArgsAmount)>
     struct UseBoundArg;
   
     CVType& cv;
   
     data::DatasetMapper<data::IncrementPolicy, double> datasetInfo;
   
     BoundArgsTupleType boundArgs;
   
     double bestObjective;
   
     MLAlgorithm bestModel;
   
     double relativeDelta;
   
     double minDelta;
   
     template<size_t BoundArgIndex,
              size_t ParamIndex,
              typename... Args,
              typename = typename
                  std::enable_if<(BoundArgIndex + ParamIndex < TotalArgs)>::type>
     inline double Evaluate(const arma::mat& parameters, const Args&... args);
   
     template<size_t BoundArgIndex,
              size_t ParamIndex,
              typename... Args,
              typename = typename
                  std::enable_if<BoundArgIndex + ParamIndex == TotalArgs>::type,
              typename = void>
     inline double Evaluate(const arma::mat& parameters, const Args&... args);
   
     template<size_t BoundArgIndex,
              size_t ParamIndex,
              typename... Args,
              typename = typename std::enable_if<
                  UseBoundArg<BoundArgIndex, ParamIndex>::value>::type>
     inline double PutNextArg(const arma::mat& parameters, const Args&... args);
   
     template<size_t BoundArgIndex,
              size_t ParamIndex,
              typename... Args,
              typename = typename std::enable_if<
                  !UseBoundArg<BoundArgIndex, ParamIndex>::value>::type,
              typename = void>
     inline double PutNextArg(const arma::mat& parameters, const Args&... args);
   };
   
   
   } // namespace hpt
   } // namespace mlpack
   
   // Include implementation
   #include "cv_function_impl.hpp"
   
   #endif
