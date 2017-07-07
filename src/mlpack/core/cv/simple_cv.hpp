/**
 * @file simple_cv.hpp
 * @author Kirill Mishchenko
 *
 * A simple cross-validation strategy.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_SIMPLE_CV_HPP
#define MLPACK_CORE_CV_SIMPLE_CV_HPP

#include <mlpack/core/cv/meta_info_extractor.hpp>
#include <mlpack/core/cv/cv_base.hpp>

namespace mlpack {
namespace cv {

/**
 * The class SimpleCV splits data into training and validation sets, runs
 * training on the training set and evaluates performance on the validation set.
 *
 * @tparam MLAlgorithm A machine learning algorithm.
 * @tparam Metric A metric to assess the quality of a trained model.
 * @tparam MatType The type of data.
 * @tparam PredictionsType The type of predictions (should be passed when the
 *     predictions type is a template parameter in Train methods of
 *     MLAlgorithm).
 * @tparam WeightsType The type of weights (should be passed when weighted
 *     learning is supported, and the weights type is a template parameter in
 *     Train methods of MLAlgorithm).
 */
template<typename MLAlgorithm,
         typename Metric,
         typename MatType = arma::mat,
         typename PredictionsType =
             typename MetaInfoExtractor<MLAlgorithm, MatType>::PredictionsType,
         typename WeightsType =
             typename MetaInfoExtractor<MLAlgorithm, MatType,
                 PredictionsType>::WeightsType>
class SimpleCV :
    private CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>
{
 public:
  /**
   * This constructor splits data into training and validation sets.
   *
   * @param validationSize A proportion (between 0 and 1) of the data used as a
   *     validation set.
   * @param args Basic constructor arguments for MLAlgortithm (see the CVBase
   *     constructors for reference).
   */
  template<typename... CVBaseArgs>
  SimpleCV(const float validationSize, const CVBaseArgs&... args);

  /**
   * Train on the training set and assess performance on the validation set by
   * using the class Metric.
   *
   * @param args Arguments for MLAlgorithm (in addition to the passed
   *     ones in the constructor).
   */
  template<typename... MLAlgorithmArgs>
  double Evaluate(const MLAlgorithmArgs& ...args);

  //! Access and modify the last trained model.
  MLAlgorithm& Model();

 private:
  //! A short alias for CVBase.
  using Base = CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>;

  /**
   * Variables for storing the whole dataset.
   */
  MatType xs;
  PredictionsType ys;
  WeightsType weights;

  /**
   * Variables for storing the training dataset.
   */
  MatType trainingXs;
  PredictionsType trainingYs;
  WeightsType trainingWeights;

  /**
   * Variables for storing the validation dataset.
   */
  MatType validationXs;
  PredictionsType validationYs;

  //! A pointer to the last trained model.
  std::unique_ptr<MLAlgorithm> modelPtr;

  /**
   * Initialize without weights.
   */
  template<typename DataArgsTupleT,
           typename = typename std::enable_if<
               std::tuple_size<DataArgsTupleT>::value == 2>::type>
  void Init(const float validationSize, const DataArgsTupleT& dataArgsTuple);

  /**
   * Initialize with weights.
   */
  template<typename DataArgsTupleT,
           typename = typename std::enable_if<
               std::tuple_size<DataArgsTupleT>::value == 3>::type,
           typename = void>
  void Init(const float validationSize, const DataArgsTupleT& dataArgsTuple);

  /**
   * Calculate the number of training points and assert it is legitimate.
   */
  size_t CalculateAndAssertNumberOfTrainingPoints(const float validationSize,
                                                  const size_t total);

  /**
   * Initialize training and validation sets.
   */
  void InitTrainingAndValidationSets(const size_t numberOfTrainingPoints);

  /**
   * Train and run evaluation in the case of non-weighted learning.
   */
  template<typename...MLAlgorithmArgs,
           bool Enabled = !Base::MIE::SupportsWeights,
           typename = typename std::enable_if<Enabled>::type>
  double TrainAndEvaluate(const MLAlgorithmArgs& ...mlAlgorithmArgs);

  /**
   * Train and run evaluation in the case of supporting weighted learning.
   */
  template<typename...MLAlgorithmArgs,
           bool Enabled = Base::MIE::SupportsWeights,
           typename = typename std::enable_if<Enabled>::type,
           typename = void>
  double TrainAndEvaluate(const MLAlgorithmArgs& ...mlAlgorithmArgs);

  /**
   * Get the specified submatrix without coping the data.
   */
  template<typename ElementType>
  arma::Mat<ElementType> GetSubset(arma::Mat<ElementType>& m,
                                   const size_t firstCol,
                                   const size_t lastCol)
  {
    return arma::Mat<ElementType>(m.colptr(firstCol), m.n_rows,
        lastCol - firstCol + 1, false, true);
  }

  /**
   * Get the specified subrow without coping the data.
   */
  template<typename ElementType>
  arma::Row<ElementType> GetSubset(arma::Row<ElementType>& r,
                                   const size_t firstCol,
                                   const size_t lastCol)
  {
    return arma::Row<ElementType>(r.colptr(firstCol), lastCol - firstCol + 1,
        false, true);
  }
};

} // namespace cv
} // namespace mlpack

// Include implementation
#include "simple_cv_impl.hpp"

#endif
