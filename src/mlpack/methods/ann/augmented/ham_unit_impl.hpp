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
    memory(TreeMemory<double, J, S>(memorySize, memoryDim, join, write))
{
  this->t = 0;
}

template<typename E, typename J, typename S, typename W, typename C>
arma::vec HAMUnit<E, J, S, W, C>::Attention()
{
  size_t nodesCnt = 2 * memory.ActualMemorySize() - 1;
  arma::vec probabilities(nodesCnt);
  probabilities(0) = 1;
  arma::vec hController = sequence.col(t);
  for (size_t node = 1; node < nodesCnt; ++node) {
    size_t parent = memory.Parent(node);
    bool dir = node == memory.Left(parent);
    arma::vec h(hController.n_elem + memory.Cell(parent).n_elem);
    h.rows(0, hController.n_elem - 1) = hController;
    h.rows(hController.n_elem,
           hController.n_elem + memory.Cell(parent).n_elem - 1)
        = memory.Cell(parent);
    arma::mat searchOutput;
    search.Predict(h, searchOutput);
    double prob = searchOutput.at(0, 0);
    if (!dir) prob = 1. - prob;
    probabilities(node) = prob * probabilities(parent);
  }
  size_t from = memory.LeafIndex(0),
         to = memory.LeafIndex(memory.MemorySize() - 1);
  arma::vec leafAttention = probabilities.rows(from, to);
  assert(abs(arma::accu(leafAttention) - 1) < 1e-4);
  return leafAttention;
}

template<typename E, typename J, typename S, typename W, typename C>
void HAMUnit<E, J, S, W, C>::Predict(arma::mat&& input, arma::mat&& output) {
  embed.Predict(input, sequence);
  memory.Initialize(sequence);
  output = arma::zeros(input.n_cols, 1);
  for (t = 0; t < input.n_cols; ++t)
  {
    arma::vec attention = Attention();
    arma::vec input = arma::zeros(memory.Leaf(0).n_elem);
    for (size_t i = 0; i < memorySize; ++i)
    {
      input += attention.at(i) * memory.Leaf(i);
    }
    arma::vec controllerOutput;
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

} // namespace augmented
} // namespace ann
} // namespace mlpack

#endif