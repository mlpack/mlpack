/**
 * @file ham_unit_impl.hpp
 * @author Konstantin Sidorov
 *
 * Implementation of HAMUnit class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_AUGMENTED_HAM_UNIT_IMPL_HPP
#define MLPACK_METHODS_ANN_AUGMENTED_HAM_UNIT_IMPL_HPP

#include "ham_unit.hpp"

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {

template<typename E, typename J, typename S, typename W, typename C>
HAMUnit<E, J, S, W, C>::HAMUnit(size_t memorySize,
                 size_t memoryDim,
                 E& embed,
                 J& join,
                 S& search,
                 W& write,
                 C& controller)
  : memorySize(memorySize), memoryDim(memoryDim),
    search(search), embed(embed), controller(controller),
    memory(TreeMemory<double, J, S>(memorySize, memoryDim, join, write)),
    t(0)
{
  // Nothing to do here
}

template<typename E, typename J, typename S, typename W, typename C>
void HAMUnit<E, J, S, W, C>::Attention(arma::vec& leafAttention)
{
  size_t nodesCnt = 2 * memory.ActualMemorySize() - 1;
  arma::vec probabilities(nodesCnt);
  probabilities(0) = 1;
  arma::vec hController = sequence.col(t);
  arma::vec h(hController.n_elem + memory.Cell(0).n_elem);
  h.rows(0, hController.n_elem - 1) = hController;
  for (size_t node = 1; node < nodesCnt; ++node) {
    size_t parent = memory.Parent(node);
    bool dir = node == memory.Left(parent);
    h.rows(hController.n_elem,
           hController.n_elem + memory.Cell(parent).n_elem - 1)
        = memory.Cell(parent);
    arma::mat searchOutput;
    search.Predict(h, searchOutput);
    double prob = searchOutput(0);
    if (!dir) prob = 1. - prob;
    probabilities(node) = prob * probabilities(parent);
  }
  size_t from = memory.LeafIndex(0),
         to = memory.LeafIndex(memory.MemorySize() - 1);
  leafAttention = probabilities.rows(from, to);
  assert(abs(arma::accu(leafAttention) - 1) < 1e-4);
}

template<typename E, typename J, typename S, typename W, typename C>
void HAMUnit<E, J, S, W, C>::Forward(arma::mat&& input, arma::mat&& output) {
  embed.Predict(input, sequence);
  memory.Initialize(sequence);
  output = arma::zeros(input.n_cols, 1);
  for (t = 0; t < input.n_cols; ++t)
  {
    arma::vec attention;
    Attention(attention);
    arma::vec input = attention.at(0) * memory.Leaf(0);
    for (size_t i = 1; i < memorySize; ++i)
    {
      input += attention.at(i) * memory.Leaf(i);
    }
    arma::rowvec controllerOutput;
    controller.Predict(input, controllerOutput);
    output.row(t) = controllerOutput;

    // Write phase.
    for (size_t i = 0; i < memorySize; ++i)
    {
      arma::vec prevMemory = memory.Leaf(i);
      memory.Update(i, input);
      memory.Leaf(i) *= attention.at(i);
      memory.Leaf(i) += (1. - attention.at(i)) * prevMemory;
    }
  }
}

template<typename E, typename J, typename S, typename W, typename C>
void HAMUnit<E, J, S, W, C>::RebuildParameters() {
  arma::mat embedParams = embed.Parameters();
  arma::mat searchParams = search.Parameters();
  arma::mat controllerParams = controller.Parameters();
  arma::mat joinParams = memory.JoinObject().Parameters();
  arma::mat writeParams = memory.WriteObject().Parameters();
  size_t embedCount = embedParams.n_elem,
         searchCount = searchParams.n_elem,
         controllerCount = controllerParams.n_elem,
         joinCount = joinParams.n_elem,
         writeCount = writeParams.n_elem,
  parameters = arma::mat(
      embedCount + searchCount + controllerCount + joinCount + writeCount, 1);
  parameters.rows(0, embedCount - 1) = embedParams;
  parameters.rows(embedCount, embedCount + searchCount - 1) = searchParams;
  parameters.rows(
      embedCount + searchCount, embedCount + searchCount + controllerCount - 1)
      = controllerParams;
  parameters.rows(searchCount, embedCount + searchCount + controllerCount,
      searchCount, embedCount + searchCount + controllerCount + joinCount - 1)
      = joinParams;
  parameters.rows(
      searchCount, embedCount + searchCount + controllerCount + joinCount,
      parameters.n_elem - 1) = writeParams;
  std::cerr << "E:\n" << embedParams;
  std::cerr << "S:\n" << searchParams;
  std::cerr << "C:\n" << controllerParams;
  std::cerr << "J:\n" << joinParams;
  std::cerr << "W:\n" << writeParams;
  std::cerr << "Total:\n" << parameters;
}

template<typename E, typename J, typename S, typename W, typename C>
void HAMUnit<E, J, S, W, C>::ResetParameters() {
  search.ResetParameters();
  embed.ResetParameters();
  controller.ResetParameters();
  memory.ResetParameters();
  RebuildParameters();
}

} // namespace augmented
} // namespace ann
} // namespace mlpack

#endif