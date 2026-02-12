/**
 * @file methods/tsne/tsne_impl.hpp
 * @author Kiner Shah
 *
 * Implementation of t-SNE class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_IMPL_HPP
#define MLPACK_METHODS_TSNE_TSNE_IMPL_HPP

// In case it hasn't been included yet.
#include "tsne.hpp"

namespace mlpack {

template<typename MatType>
TSNE<MatType>::TSNE(const double perplexity,
                    const double learningRate,
                    const size_t maxIterations,
                    const double earlyExaggeration) :
    perplexity(perplexity),
    learningRate(learningRate),
    maxIterations(maxIterations),
    earlyExaggeration(earlyExaggeration)
{
  if (perplexity <= 0.0)
  {
    Log::Fatal << "TSNE::TSNE(): perplexity must be positive!" << std::endl;
  }
  if (learningRate <= 0.0)
  {
    Log::Fatal << "TSNE::TSNE(): learning rate must be positive!" << std::endl;
  }
}

template<typename MatType>
template<typename OutMatType>
void TSNE<MatType>::Apply(const MatType& data,
                          OutMatType& output,
                          const size_t newDimension)
{
  using ElemType = typename MatType::elem_type;
  
  const size_t n = data.n_cols;
  
  if (n == 0)
  {
    Log::Fatal << "TSNE::Apply(): cannot run t-SNE on empty dataset!"
        << std::endl;
  }
  
  Log::Info << "Running t-SNE on " << n << " points in "
      << data.n_rows << " dimensions." << std::endl;
  Log::Info << "Target dimensionality: " << newDimension << std::endl;
  Log::Info << "Perplexity: " << perplexity << std::endl;
  
  // Step 1: Compute pairwise affinities P in high-dimensional space.
  MatType P;
  ComputeAffinities(data, P);
  
  // Step 2: Initialize low-dimensional embedding Y randomly.
  output.randn(newDimension, n);
  output *= 0.0001; // Small random initialization
  
  // Step 3: Optimize the embedding using gradient descent.
  MatType gradient(newDimension, n);
  MatType velocity(newDimension, n, arma::fill::zeros);
  const double momentum = 0.5;
  const double finalMomentum = 0.8;
  const size_t momentumSwitchIter = 250;
  const size_t exaggerationIter = 250;
  
  // Apply early exaggeration to P
  MatType Pexaggerated = P * earlyExaggeration;
  
  for (size_t iter = 0; iter < maxIterations; ++iter)
  {
    // Switch from early exaggeration to normal P
    const MatType& Pcurrent = (iter < exaggerationIter) ? Pexaggerated : P;
    
    // Switch momentum
    const double currentMomentum = (iter < momentumSwitchIter) ? 
        momentum : finalMomentum;
    
    // Compute gradient
    ComputeGradient(Pcurrent, output, gradient);
    
    // Update with momentum
    velocity = currentMomentum * velocity - learningRate * gradient;
    output += velocity;
    
    // Center the embedding (remove translation)
    output.each_col() -= arma::mean(output, 1);
    
    if ((iter + 1) % 100 == 0 || iter == 0)
    {
      Log::Info << "Iteration " << (iter + 1) << " / " << maxIterations
          << std::endl;
    }
  }
  
  Log::Info << "t-SNE complete." << std::endl;
}

template<typename MatType>
void TSNE<MatType>::ComputeAffinities(const MatType& data, MatType& affinities)
{
  using ElemType = typename MatType::elem_type;
  const size_t n = data.n_cols;
  
  // Compute pairwise squared Euclidean distances
  MatType squaredDistances(n, n);
  for (size_t i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < n; ++j)
    {
      if (i == j)
      {
        squaredDistances(i, j) = 0.0;
      }
      else
      {
        ElemType dist = arma::norm(data.col(i) - data.col(j));
        squaredDistances(i, j) = dist * dist;
      }
    }
  }
  
  // Compute conditional probabilities P(j|i) using binary search
  MatType conditionalP(n, n);
  for (size_t i = 0; i < n; ++i)
  {
    typename MatType::col_type distances = squaredDistances.col(i);
    typename MatType::col_type probabilities;
    SearchPrecision(distances, perplexity, probabilities);
    conditionalP.col(i) = probabilities;
  }
  
  // Symmetrize: P = (P + P^T) / (2 * n)
  affinities = (conditionalP + conditionalP.t()) / (2.0 * n);
  
  // Ensure minimum value for numerical stability
  affinities = arma::max(affinities, 1e-12);
}

template<typename MatType>
void TSNE<MatType>::SearchPrecision(
    const typename MatType::col_type& distances,
    const double perplexity,
    typename MatType::col_type& probabilities)
{
  using ElemType = typename MatType::elem_type;
  const size_t n = distances.n_elem;
  
  probabilities.set_size(n);
  
  // Binary search for precision beta that gives desired perplexity
  double betaMin = -std::numeric_limits<double>::infinity();
  double betaMax = std::numeric_limits<double>::infinity();
  double beta = 1.0;
  const double logPerplexity = std::log(perplexity);
  const size_t maxTries = 50;
  const double tolerance = 1e-5;
  
  for (size_t iter = 0; iter < maxTries; ++iter)
  {
    // Compute conditional probabilities with current beta
    typename MatType::col_type expDistances(n);
    ElemType sumExpDistances = 0.0;
    
    for (size_t j = 0; j < n; ++j)
    {
      expDistances(j) = std::exp(-beta * distances(j));
      sumExpDistances += expDistances(j);
    }
    
    // Normalize (excluding self-distance which is 0)
    ElemType sumP = 0.0;
    for (size_t j = 0; j < n; ++j)
    {
      if (sumExpDistances > 0)
        probabilities(j) = expDistances(j) / sumExpDistances;
      else
        probabilities(j) = 0.0;
      
      if (probabilities(j) > 1e-12)
        sumP += probabilities(j);
    }
    
    // Compute entropy H(P_i)
    ElemType entropy = 0.0;
    for (size_t j = 0; j < n; ++j)
    {
      if (probabilities(j) > 1e-12)
        entropy -= probabilities(j) * std::log(probabilities(j));
    }
    
    // Perplexity = 2^H
    double currentPerplexity = std::exp(entropy);
    double perplexityDiff = currentPerplexity - perplexity;
    
    if (std::abs(perplexityDiff) < tolerance)
      break;
    
    // Adjust beta using binary search
    if (perplexityDiff > 0)
    {
      betaMin = beta;
      if (std::isinf(betaMax))
        beta *= 2.0;
      else
        beta = (beta + betaMax) / 2.0;
    }
    else
    {
      betaMax = beta;
      if (std::isinf(betaMin))
        beta /= 2.0;
      else
        beta = (beta + betaMin) / 2.0;
    }
  }
}

template<typename MatType>
void TSNE<MatType>::ComputeGradient(const MatType& P,
                                     const MatType& Y,
                                     MatType& gradient)
{
  using ElemType = typename MatType::elem_type;
  const size_t n = Y.n_cols;
  const size_t d = Y.n_rows;
  
  // Compute pairwise squared distances in low-dimensional space
  MatType squaredDistances(n, n);
  for (size_t i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < n; ++j)
    {
      if (i == j)
      {
        squaredDistances(i, j) = 0.0;
      }
      else
      {
        ElemType dist = arma::norm(Y.col(i) - Y.col(j));
        squaredDistances(i, j) = dist * dist;
      }
    }
  }
  
  // Compute Q using Student t-distribution with 1 degree of freedom
  MatType Q(n, n);
  ElemType sumQ = 0.0;
  for (size_t i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < n; ++j)
    {
      if (i != j)
      {
        Q(i, j) = 1.0 / (1.0 + squaredDistances(i, j));
        sumQ += Q(i, j);
      }
      else
      {
        Q(i, j) = 0.0;
      }
    }
  }
  
  // Normalize Q
  if (sumQ > 0)
    Q /= sumQ;
  
  // Ensure minimum value
  Q = arma::max(Q, 1e-12);
  
  // Compute gradient
  gradient.zeros(d, n);
  for (size_t i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < n; ++j)
    {
      if (i != j)
      {
        ElemType multiplier = (P(i, j) - Q(i, j)) * Q(i, j) * (1.0 + squaredDistances(i, j));
        gradient.col(i) += multiplier * (Y.col(i) - Y.col(j));
      }
    }
    gradient.col(i) *= 4.0; // Factor from the derivative
  }
}

} // namespace mlpack

#endif
