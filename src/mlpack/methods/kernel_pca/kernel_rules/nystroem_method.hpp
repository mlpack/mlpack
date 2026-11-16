/**
 * @file methods/kernel_pca/kernel_rules/nystroem_method.hpp
 * @author Marcus Edel
 *
 * Use the Nystroem method for approximating a kernel matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_KERNEL_PCA_KERNEL_RULES_NYSTROEM_METHOD_HPP
#define MLPACK_METHODS_KERNEL_PCA_KERNEL_RULES_NYSTROEM_METHOD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/nystroem_method/kmeans_selection.hpp>
#include <mlpack/methods/nystroem_method/nystroem_method.hpp>

namespace mlpack {

template<
  typename PointSelectionPolicy = KMeansSelection<>
>
class NystroemKernelRule
{
 public:
  /**
   * Construct the NystroemKernelRule object, using the given percentage of the
   * dataset size as the number of samples.
   */
  template<typename T = double>
  NystroemKernelRule(
      const T datasetPctIn = T(0.1),
      const typename std::enable_if_t<std::is_floating_point_v<T>>* = 0) :
      numPoints(0),
      datasetPct(double(datasetPctIn)),
      usingNumPoints(false)
  {
    if (datasetPct <= 0.0)
    {
      throw std::invalid_argument("NystroemKernelRule::NystroemKernelRule(): "
          "dataset percentage must be greater than 0!");
    }
    else if (datasetPct > 1.0)
    {
      throw std::invalid_argument("NystroemKernelRule::NystroemKernelRule(): "
          "dataset percentage must be less than or equal to 1!");
    }
  }

  /*
   * Construct the NystroemKernelRule object, using `numPoints` points to sample
   * the dataset.
   */
  template<typename T>
  NystroemKernelRule(
      const T numPoints,
      const typename std::enable_if_t<std::is_integral_v<T>>* = 0) :
      numPoints(numPoints),
      datasetPct(0.0),
      usingNumPoints(true) { }

  /**
   * Construct the kernel matrix approximation using the nystroem method.
   *
   * @param data Input data points.
   * @param transformedData Matrix to output results into.
   * @param eigval KPCA eigenvalues will be written to this vector.
   * @param eigvec KPCA eigenvectors will be written to this matrix.
   * @param rank Rank to be used for matrix approximation.
   * @param kernel Kernel to be used for computation.
   */
  template<typename MatType, typename VecType, typename KernelType>
  void ApplyKernelMatrix(const MatType& data,
                         MatType& transformedData,
                         VecType& eigval,
                         MatType& eigvec,
                         KernelType kernel = KernelType())
  {
    const size_t points = usingNumPoints ?
        std::min(numPoints, size_t(data.n_cols)) :
        size_t(datasetPct * data.n_cols);

    MatType G, v;
    NystroemMethod<KernelType, PointSelectionPolicy, MatType> nm(points,
        kernel);
    nm.Apply(data, G);
    transformedData = G.t() * G;

    // Center the reconstructed approximation.
    transformedData.each_col() -= mean(transformedData, 1);

    // For PCA the data has to be centered, even if the data is centered. But
    // it is not guaranteed that the data, when mapped to the kernel space, is
    // also centered. Since we actually never work in the feature space we
    // cannot center the data. So, we perform a "psuedo-centering" using the
    // kernel matrix.
    typename GetColType<MatType>::type colMean = sum(G, 1) / G.n_rows;
    G.each_row() -= sum(G, 0) / G.n_rows;
    G.each_col() -= colMean;
    G += sum(colMean) / G.n_rows;

    // Eigendecompose the centered kernel matrix.
    transformedData = symmatu(transformedData);
    if (!eig_sym(eigval, eigvec, transformedData))
    {
      Log::Fatal << "Failed to construct the kernel matrix." << std::endl;
    }

    // Swap the eigenvalues since they are ordered backwards (we need largest
    // to smallest).
    for (size_t i = 0; i < floor(eigval.n_elem / 2.0); ++i)
      eigval.swap_rows(i, (eigval.n_elem - 1) - i);

    // Flip the coefficients to produce the same effect.
    eigvec = fliplr(eigvec);

    transformedData = eigvec.t() * G.t();
  }

  // Serialize the NystroemKernelRule.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar(CEREAL_NVP(numPoints));
    ar(CEREAL_NVP(datasetPct));
    ar(CEREAL_NVP(usingNumPoints));
  }

 private:
  // The number of points to use from the dataset.
  size_t numPoints;
  // The percentage of the dataset to use as the sample.
  double datasetPct;
  // Whether we are using the number of points or dataset percentage.
  bool usingNumPoints;
};

} // namespace mlpack

#endif
