/**
 * @file k_fold_cv.hpp
 * @author Kirill Mishchenko
 *
 * k-fold cross-validation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_K_FOLD_CV_HPP
#define MLPACK_CORE_CV_K_FOLD_CV_HPP

#include <mlpack/core/cv/meta_info_extractor.hpp>
#include <mlpack/core/cv/cv_base.hpp>

namespace mlpack {
namespace cv {

/**
 * The class KFoldCV implements k-fold cross-validation for regression and
 * classification algorithms.
 *
 * To construct a KFoldCV object you need to pass the k parameter and arguments
 * that specify data. For the latter see the CVBase constructors as a reference
 * - the CVBase constructors take exactly the same arguments as ones that are
 * supposed to be passed after the k parameter in the KFoldCV constructor.
 *
 * For example, you can run 10-fold cross-validation for SoftmaxRegression in
 * the following way.
 *
 * @code
 * // 100-point 5-dimensional random dataset.
 * arma::mat data = arma::randu<arma::mat>(5, 100);
 * // Random labels in the [0, 4] interval.
 * arma::Row<size_t> labels =
 *     arma::randi<arma::Row<size_t>>(100, arma::distr_param(0, 4));
 * size_t numClasses = 5;
 *
 * KFoldCV<SoftmaxRegression<>, Accuracy> cv(10, data, labels, numClasses);
 *
 * double lambda = 0.1;
 * double softmaxAccuracy = cv.Evaluate(lambda);
 * @endcode
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
class KFoldCV :
    private CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>
{
 public:
  /**
   * Construct an object for running k-fold cross-validation.
   *
   * @param k Number of folds (should be at least 2).
   * @param args Basic constructor arguments for MLAlgortithm (see the CVBase
   *     constructors for reference).
   */
  template<typename... CVBaseArgs>
  KFoldCV(const size_t k, const CVBaseArgs&... args);

  /**
   * Run k-fold cross-validation.
   *
   * @param args Arguments for MLAlgorithm (in addition to the passed
   *     ones in the constructor).
   */
  template<typename... MLAlgorithmArgs>
  double Evaluate(const MLAlgorithmArgs& ...args);

  //! Access and modify a model from the last run of k-fold cross-validation.
  MLAlgorithm& Model();

 private:
  //! A short alias for CVBase.
  using Base = CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>;

  //! The number of bins in the dataset.
  const size_t k;

  /**
   * Variables for storing the extended (by repeating the first k - 2 bins)
   * dataset.
   */
  MatType xs;
  PredictionsType ys;
  WeightsType weights;

  //! The size of each bin in terms of data points.
  size_t binSize;

  //! The size of each training subset in terms of data points.
  size_t trainingSubsetSize;

  //! A pointer to a model from the last run of k-fold cross-validation.
  std::unique_ptr<MLAlgorithm> modelPtr;

  /**
   * Initialize without weights.
   */
  template<typename DataArgsTupleT,
           typename = typename std::enable_if<
               std::tuple_size<DataArgsTupleT>::value == 2>::type>
  void Init(const DataArgsTupleT& dataArgsTuple);

  /**
   * Initialize with weights.
   */
  template<typename DataArgsTupleT,
           typename = typename std::enable_if<
               std::tuple_size<DataArgsTupleT>::value == 3>::type,
           typename = void>
  void Init(const DataArgsTupleT& dataArgsTuple);

  /**
   * Initialize the given destination matrix with the given source joined with
   * its first k - 2 bins.
   */
  template<typename SourceType, typename DestinationType>
  void InitKFoldCVMat(const SourceType& source, DestinationType& destination);

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
   * Calculate the index of the first column of the ith validation subset.
   *
   * We take the ith validation subset after the ith training subset if
   * i < k - 1 and before it otherwise.
   */
  size_t ValidationSubsetFirtsCol(const size_t i)
  {
    return i < k - 1? binSize * i + trainingSubsetSize : binSize * (i - 1);
  }

  /**
   * Get the ith training subset from a variable of a matrix type.
   */
  template<typename ElementType>
  arma::Mat<ElementType> GetTrainingSubset(arma::Mat<ElementType>& m,
                                           const size_t i)
  {
    return arma::Mat<ElementType>(m.colptr(binSize * i), m.n_rows,
        trainingSubsetSize, false, true);
  }

  /**
   * Get the ith training subset from a variable of a row type.
   */
  template<typename ElementType>
  arma::Row<ElementType> GetTrainingSubset(arma::Row<ElementType>& r,
                                           const size_t i)
  {
    return arma::Row<ElementType>(r.colptr(binSize * i), trainingSubsetSize,
        false, true);
  }

  /**
   * Get the ith validation subset from a variable of a matrix type.
   */
  template<typename ElementType>
  arma::Mat<ElementType> GetValidationSubset(arma::Mat<ElementType>& m,
                                             const size_t i)
  {
    return arma::Mat<ElementType>(m.colptr(ValidationSubsetFirtsCol(i)),
        m.n_rows, binSize, false, true);
  }

  /**
   * Get the ith validation subset from a variable of a row type.
   */
  template<typename ElementType>
  arma::Row<ElementType> GetValidationSubset(arma::Row<ElementType>& r,
                                             const size_t i)
  {
    return arma::Row<ElementType>(r.colptr(ValidationSubsetFirtsCol(i)),
        binSize, false, true);
  }

};

} // namespace cv
} // namespace mlpack

// Include implementation
#include "k_fold_cv_impl.hpp"

#endif
