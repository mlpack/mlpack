/**
 * @file methods/preprocess/scaling_model.hpp
 * @author Jeffin Sam
 *
 * A serializable Scaling model, used by the main program.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
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

/**
 * The model to save to disk.
 */
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
  //! Create an object.
  ScalingModel(const int minvalue = 0, const int maxvalue = 1,
      double epsilonvalue = 0.00005);

  //! Copy constructor.
  ScalingModel(const ScalingModel& other);

  //! Move constructor.
  ScalingModel(ScalingModel&& other);

  //! Copy assignment operator.
  ScalingModel& operator=(const ScalingModel& other);

  //! Move assignment operator.
  ScalingModel& operator=(ScalingModel&& other);

  //! Clean up memory.
  ~ScalingModel();

  //! Get the Scaler type.
  size_t ScalerType() const { return scalerType; }
  //! Modify the Scaler type.
  size_t& ScalerType() { return scalerType; }

  //! Transform to scale features.
  template<typename MatType>
  void Transform(const MatType& input, MatType& output);

  // Fit to intialize the scaling parameter.
  template<typename MatType>
  void Fit(const MatType& input);

  // Scale back the dataset to their original values.
  template<typename MatType>
  void InverseTransform(const MatType& input, MatType& output);

  //! Serialize the model.
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
