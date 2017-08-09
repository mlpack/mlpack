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

HAMUnit::HAMUnit(size_t memorySize,
                 LayerTypes& embed,
                 LayerTypes& join,
                 LayerTypes& search,
                 LayerTypes& write)
  : memorySize(memorySize), search(search), embed(embed), t(0)
{
  memory = TreeMemory<double>(memorySize, join, write);
}

arma::vec HAMUnit::Attention() const
{
  size_t nodesCnt = 2 * memory.ActualMemorySize() - 1;
  arma::vec probabilities(nodesCnt);
  probabilities(0) = 1;
  arma::vec hController = sequence.col(t++);
  for (size_t node = 1; node < nodesCnt; ++node) {
    size_t parent = memory.Parent(node);
    bool dir = node == memory.Left(parent);
    arma::vec h(hController.n_elem + memory.Cell(parent).n_elem);
    h.rows(0, hController.n_elem - 1) = hController;
    h.rows(hController.n_elem,
           hController.n_elem + memory.Cell(parent).n_elem - 1)
        = memory.Cell(parent);
    double prob = search.Forward(h);
    if (!dir) prob = 1. - prob;
    probabilities(node) = prob * probabilities(parent);
  }
  size_t from = mem.LeafIndex(0),
         to = memory.LeafIndex(memory.MemorySize() - 1);
  arma::vec leafAttention = probabilities.rows(from, to);
  assert(abs(arma::accu(leafAttention) - 1) < 1e-4);
  return leafAttention;
}

void HAMUnit::Forward(arma::mat&& input, arma::mat&& output) {
  sequence = embed.Predict(input);
  memory.Initialize(sequence);
  for (t = 0; t < input.n_cols; ++t)
  {
    arma::vec attention = Attention();
    arma::vec input = arma::zeros(memory.Get(0).n_elem);
    for (size_t i = 0; i < memorySize; ++i)
    {
      input += leafAttention.at(i) * memory.Get(i);
    }
    output.at(t) = 0; // TODO?

    // Write phase.
    for (size_t i = 0; i < memory.MemorySize(); ++i)
    {
      arma::vec prevMemory = memory.Get(i);
      memory.Update(i, write.Forward(memory.Stack(memory.Get(i), input)))
      memory.Get(i) *= attention.at(i);
      memory.Get(i) += (1. - attention.at(i)) * prevMemory;
    }
  }
}

} // namespace augmented
} // namespace ann
} // namespace mlpack

#endif