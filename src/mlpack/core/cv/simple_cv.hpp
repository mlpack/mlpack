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
 * SimpleCV splits data into two sets - training and validation sets - and then
 * runs training on the training set and evaluates performance on the validation
 * set.
 *
 * To construct a SimpleCV object you need to pass the validationSize parameter
 * and arguments that specify data. For the latter see the CVBase constructors
 * as a reference - the CVBase constructors take exactly the same arguments as
 * ones that are supposed to be passed after the validationSize parameter in the
 * SimpleCV constructor.
 *
 * For example, SoftmaxRegression can be validated in the following way.
 *
 * arma::mat data = ...;
 * arma::Row<size_t> labels = ...;
 * size_t numClasses = 5;
 *
 * double validationSize = 0.2;
 * SimpleCV<SoftmaxRegression<>, Accuracy> cv(validationSize, data, labels,
 *     numClasses);
 *
 * double lambda = 0.1;
 * double softmaxAccuracy = cv.Evaluate(lambda);
 *
 * In the example above, 80% of the passed dataset will be used for training,
 * and remaining 20% will be used for calculating the accuracy metric.
 *
 * @tparam MLAlgorithm A machine learning algorithm.
 * @tparam Metric A metric to assess the quality of a trained model.
 * @tparam MatType The type of data.
 * @tparam PredictionsType The type of predictions (should be passed when the
 *     predictions type is a template parameter in Train methods of the given
 *     MLAlgorithm; arma::Row<size_t> will be used otherwise).
 * @tparam WeightsType The type of weights (should be passed when weighted
 *     learning is supported, and the weights type is a template parameter in
 *     Train methods of the given MLAlgorithm; arma::vec will be used
 *     otherwise).
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
  SimpleCV(const double validationSize, const CVBaseArgs&... args);

  /**
   * Train on the training set and assess performance on the validation set by
   * using the class Metric.
   *
   * @param args Arguments for the given MLAlgorithm taken by its constructor
   *     (in addition to the passed ones in the SimpleCV constructor).
   */
  template<typename... MLAlgorithmArgs>
  double Evaluate(const MLAlgorithmArgs&... args);

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
  void Init(const double validationSize, const DataArgsTupleT& dataArgsTuple);

  /**
   * Initialize with weights.
   */
  template<typename DataArgsTupleT,
           typename = typename std::enable_if<
               std::tuple_size<DataArgsTupleT>::value == 3>::type,
           typename = void>
  void Init(const double validationSize, const DataArgsTupleT& dataArgsTuple);

  /**
   * Initialize training and validation sets.
   */
  void InitTrainingAndValidationSets(const double validationSize);

  /**
   * Calculate the number of training points and assert it is legitimate.
   */
  size_t CalculateAndAssertNumberOfTrainingPoints(const double validationSize);

  /**
   * Train and run evaluation in the case of non-weighted learning.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = !Base::MIE::SupportsWeights,
           typename = typename std::enable_if<Enabled>::type>
  double TrainAndEvaluate(const MLAlgorithmArgs&... args);

  /**
   * Train and run evaluation in the case of supporting weighted learning.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = Base::MIE::SupportsWeights,
           typename = typename std::enable_if<Enabled>::type,
           typename = void>
  double TrainAndEvaluate(const MLAlgorithmArgs&... args);

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
