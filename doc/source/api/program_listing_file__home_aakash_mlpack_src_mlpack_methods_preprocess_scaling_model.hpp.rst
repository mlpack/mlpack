
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_preprocess_scaling_model.hpp:

Program Listing for File scaling_model.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_preprocess_scaling_model.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/preprocess/scaling_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_SCALING_MODEL_HPP
   #define MLPACK_CORE_DATA_SCALING_MODEL_HPP
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/data/scaler_methods/max_abs_scaler.hpp>
   #include <mlpack/core/data/scaler_methods/mean_normalization.hpp>
   #include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
   #include <mlpack/core/data/scaler_methods/pca_whitening.hpp>
   #include <mlpack/core/data/scaler_methods/zca_whitening.hpp>
   #include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
   
   namespace mlpack {
   namespace data {
   
   class ScalingModel
   {
    public:
     enum ScalerTypes
     {
       STANDARD_SCALER,
       MIN_MAX_SCALER,
       MEAN_NORMALIZATION,
       MAX_ABS_SCALER,
       PCA_WHITENING,
       ZCA_WHITENING
     };
   
    private:
     size_t scalerType;
     data::MinMaxScaler* minmaxscale;
     data::MaxAbsScaler* maxabsscale;
     data::MeanNormalization* meanscale;
     data::StandardScaler* standardscale;
     data::PCAWhitening* pcascale;
     data::ZCAWhitening* zcascale;
     int minValue;
     int maxValue;
     double epsilon;
   
    public:
     ScalingModel(const int minvalue = 0, const int maxvalue = 1,
         double epsilonvalue = 0.00005);
   
     ScalingModel(const ScalingModel& other);
   
     ScalingModel(ScalingModel&& other);
   
     ScalingModel& operator=(const ScalingModel& other);
   
     ScalingModel& operator=(ScalingModel&& other);
   
     ~ScalingModel();
   
     size_t ScalerType() const { return scalerType; }
     size_t& ScalerType() { return scalerType; }
   
     template<typename MatType>
     void Transform(const MatType& input, MatType& output);
   
     // Fit to intialize the scaling parameter.
     template<typename MatType>
     void Fit(const MatType& input);
   
     // Scale back the dataset to their original values.
     template<typename MatType>
     void InverseTransform(const MatType& input, MatType& output);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       if (cereal::is_loading<Archive>())
       {
         if (minmaxscale)
           delete minmaxscale;
         if (maxabsscale)
           delete maxabsscale;
         if (meanscale)
           delete meanscale;
         if (standardscale)
           delete standardscale;
         if (pcascale)
           delete pcascale;
         if (zcascale)
           delete zcascale;
   
         minmaxscale = NULL;
         maxabsscale = NULL;
         standardscale = NULL;
         meanscale = NULL;
         pcascale = NULL;
         zcascale = NULL;
       }
   
       ar(CEREAL_NVP(scalerType));
       ar(CEREAL_NVP(epsilon));
       ar(CEREAL_NVP(minValue));
       ar(CEREAL_NVP(maxValue));
       if (scalerType == ScalerTypes::MIN_MAX_SCALER)
         ar(CEREAL_POINTER(minmaxscale));
       else if (scalerType == ScalerTypes::MEAN_NORMALIZATION)
         ar(CEREAL_POINTER(meanscale));
       else if (scalerType == ScalerTypes::MAX_ABS_SCALER)
         ar(CEREAL_POINTER(maxabsscale));
       else if (scalerType == ScalerTypes::STANDARD_SCALER)
         ar(CEREAL_POINTER(standardscale));
       else if (scalerType == ScalerTypes::PCA_WHITENING)
         ar(CEREAL_POINTER(pcascale));
       else if (scalerType == ScalerTypes::ZCA_WHITENING)
         ar(CEREAL_POINTER(zcascale));
     }
   };
   
   } // namespace data
   } // namespace mlpack
   
   // Include implementation.
   #include "scaling_model_impl.hpp"
   
   #endif
