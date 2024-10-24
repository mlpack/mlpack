/**
 * @file methods/ann/convolution_rules/svd_convolution.hpp
 * @author Marcus Edel
 *
 * Implementation of the convolution using the singular value decomposition to
 * speed up the computation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_CONVOLUTION_RULES_SVD_CONVOLUTION_HPP
#define MLPACK_METHODS_ANN_CONVOLUTION_RULES_SVD_CONVOLUTION_HPP

#include <mlpack/prereqs.hpp>
#include "border_modes.hpp"
#include "fft_convolution.hpp"
#include "naive_convolution.hpp"

namespace mlpack {

/**
 * Computes the two-dimensional convolution using singular value decomposition.
 * This class allows specification of the type of the border type. The
 * convolution can be computed with the valid border type of the full border
 * type (default).
 *
 * FullConvolution: returns the full two-dimensional convolution.
 * ValidConvolution: returns only those parts of the convolution that are
 * computed without the zero-padded edges.
 *
 * @tparam BorderMode Type of the border mode (FullConvolution or
 * ValidConvolution).
 */
template<typename BorderMode = FullConvolution>
class SVDConvolution
{
 public:
  /**
   * Perform a convolution (valid or full mode) using singular value
   * decomposition. By using singular value decomposition of the filter matrix
   * the convolution can be expressed as a sum of outer products. Each product
   * can be computed efficiently as convolution with a row and a column vector.
   * The individual convolutions are computed with the naive implementation
   * which is fast if the filter is low-dimensional.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename MatType>
  static void Convolution(
      const MatType& input,
      const MatType& filter,
      MatType& output,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    using ColType = typename GetColType<MatType>::type;
    // Use the naive convolution in case the filter isn't two dimensional or the
    // filter is bigger than the input.
    if (filter.n_rows > input.n_rows || filter.n_cols > input.n_cols ||
        filter.n_rows == 1 || filter.n_cols == 1)
    {
      NaiveConvolution<BorderMode>::Convolution(input, filter, output);
    }
    else
    {
      MatType U, V, subOutput;
      ColType s;

      svd_econ(U, s, V, filter);

      // Rank approximation using the singular values calculated with singular
      // value decomposition of dense filter matrix.
      const size_t rank = sum(s > (s.n_elem * max(s) * arma::datum::eps));

      // Test for separability based on the rank of the kernel and take
      // advantage of the low rank.
      if (rank * (filter.n_rows + filter.n_cols) < filter.n_elem)
      {
        MatType subFilter = V.unsafe_col(0) * s(0);
        NaiveConvolution<BorderMode>::Convolution(input, subFilter, subOutput);

        subOutput = subOutput.t();
        NaiveConvolution<BorderMode>::Convolution(subOutput, U.unsafe_col(0),
            output);

        MatType temp;
        for (size_t r = 1; r < rank; r++)
        {
          subFilter = V.unsafe_col(r) * s(r);
          NaiveConvolution<BorderMode>::Convolution(input, subFilter,
              subOutput);

          subOutput = subOutput.t();
          NaiveConvolution<BorderMode>::Convolution(subOutput, U.unsafe_col(r),
              temp);
          output += temp;
        }

        output = output.t();
      }
      else
      {
        FFTConvolution<BorderMode>::Convolution(input, filter, output);
      }
    }
  }

  /**
   * Perform a convolution using 3rd order tensors.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename CubeType>
  static void Convolution(
      const CubeType& input,
      const CubeType& filter,
      CubeType& output,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    using MatType = typename GetDenseMatType<CubeType>::type;
    MatType convOutput;
    SVDConvolution<BorderMode>::Convolution(input.slice(0), filter.slice(0),
        convOutput);

    output = CubeType(convOutput.n_rows, convOutput.n_cols, input.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < input.n_slices; ++i)
    {
      SVDConvolution<BorderMode>::Convolution(input.slice(i), filter.slice(i),
          output.slice(i));
    }
  }

  /**
   * Perform a convolution using dense matrix as input and a 3rd order tensors
   * as filter and output.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename MatType, typename CubeType>
  static void Convolution(
      const MatType& input,
      const CubeType& filter,
      CubeType& output,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    MatType convOutput;
    SVDConvolution<BorderMode>::Convolution(input, filter.slice(0), convOutput);

    output = CubeType(convOutput.n_rows, convOutput.n_cols, filter.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < filter.n_slices; ++i)
    {
      SVDConvolution<BorderMode>::Convolution(input, filter.slice(i),
          output.slice(i));
    }
  }

  /**
   * Perform a convolution using a 3rd order tensors as input and output and a
   * dense matrix as filter.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename MatType, typename CubeType>
  static void Convolution(
      const CubeType& input,
      const MatType& filter,
      CubeType& output,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    MatType convOutput;
    SVDConvolution<BorderMode>::Convolution(input.slice(0), filter, convOutput);

    output = CubeType(convOutput.n_rows, convOutput.n_cols, input.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < input.n_slices; ++i)
    {
      SVDConvolution<BorderMode>::Convolution(input.slice(i), filter,
          output.slice(i));
    }
  }
};  // class SVDConvolution

} // namespace mlpack

#endif
