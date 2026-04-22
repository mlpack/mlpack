/**
 * @file core/transforms/eemd.hpp
 * @author Mohammad Mundiwala
 *
 * Implementation of the EEMD feature extractor (ensemble empirical mode decomposition)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TRANSFORMS_EEMD_HPP
#define MLPACK_CORE_TRANSFORMS_EEMD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/transforms/spline_envelope.hpp>
#include <mlpack/core/transforms/emd.hpp>

namespace mlpack {


/**
 * Ensemble Empirical Mode Decomposition on a 1D signal.
 * Algorithm outline: run EMD N times with added noise and average the resulting
 * IMFs.
 * Supports any Armadillo-compatible column/matrix types with matching element
 * types (enforced via SFINAE).
 *
 * @tparam ColType  Armadillo compatible column vector type.
 * @tparam MatType  Armadillo compatible matrix type.
 * @param signal Input 1D signal (length N).
 * @param imfs Output matrix of extracted IMFs, size N x K.
 * @param residue Output final residue, size N x 1.
 * @param ensembleSize Number of EMD runs to average for EEMD.
 * @param noiseStrength Strength of added noise for EEMD.
 * @param maxImfs Maximum number of IMFs to extract.
 * @param maxSiftIter Maximum number of sifting iterations per IMF.
 * @param tol Stopping tolerance used by the sifting procedure.
 */
template<typename ColType,
         typename MatType,
         typename = std::enable_if_t<
             std::is_same_v<typename ColType::elem_type,
                            typename MatType::elem_type>>>
inline void EEMD(const ColType& signal,
                MatType& imfs,
                ColType& residue,
                const size_t ensembleSize = 100,
                const double noiseStrength = 0.1,
                const size_t maxImfs = 10,
                const size_t maxSiftIter = 50,
                const double tol = 1e-3)
{
  if (signal.n_cols > 1)
    throw std::runtime_error("EEMD(): given signal must have only one column!");
  if (ensembleSize == 0)
    throw std::runtime_error("EEMD(): ensemble size must be > 0!");

  const size_t N = signal.n_elem;
  imfs.reset();
  residue = signal;

  if (N == 0)
    return;

  const double noiseScale = noiseStrength * stddev(signal);

  // Add white noise, then average IMFs over each pass
  MatType accumulatedImfs(N, maxImfs, arma::fill::zeros);
  size_t numImfs = maxImfs;

#ifdef MLPACK_USE_OPENMP
  const size_t threadCount = static_cast<size_t>(omp_get_max_threads());
#else
  const size_t threadCount = 1;
#endif

  std::vector<MatType> partialImfs(threadCount);
  for (MatType& partial : partialImfs)
    partial.zeros(N, maxImfs);

  std::vector<size_t> partialMinImfs(threadCount, maxImfs);
  // Parallelize over threads, so each thread accumulates its own imfs
  // and minImfs to avoid race conditions, if multiple threads exist.
  #pragma omp parallel for 
  for (size_t i = 0; i < ensembleSize; ++i)
  {
#ifdef MLPACK_USE_OPENMP
    const size_t threadId = static_cast<size_t>(omp_get_thread_num());
#else
    const size_t threadId = 0;
#endif

    MatType& localAccum = partialImfs[threadId];
    size_t localMin = maxImfs;
    
    ColType noisySignal = signal + noiseScale * randn<ColType>(N);
    MatType imfsNoisy;
    ColType residueNoisy;
    EMD(noisySignal, imfsNoisy, residueNoisy, maxImfs, maxSiftIter, tol);

    localMin = std::min(localMin, (size_t) imfsNoisy.n_cols);
    if (imfsNoisy.n_cols > 0)
      localAccum.cols(0, imfsNoisy.n_cols - 1) += imfsNoisy;
    
    partialMinImfs[threadId] = localMin;
  }

  for (size_t thread = 0; thread < threadCount; ++thread)
  {
    accumulatedImfs += partialImfs[thread];
    numImfs = std::min(numImfs, partialMinImfs[thread]);
  }

  imfs = accumulatedImfs.cols(0, numImfs - 1) / ensembleSize;
  residue = signal - sum(imfs, 1);
}
} // namespace mlpack

#endif // MLPACK_CORE_TRANSFORMS_EEMD_HPP
