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

template<typename E, typename J, typename S, typename W>
HAMUnit<E, J, S, W>::HAMUnit(size_t memorySize,
                 size_t memoryDim,
                 E& embed,
                 J& join,
                 S& search,
                 W& write)
  : memorySize(memorySize), memoryDim(memoryDim),
    search(search), embed(embed),
    memory(TreeMemory<double, J, S>(memorySize, memoryDim, join, write))
{
  this->t = 0;
}

template<typename E, typename J, typename S, typename W>
arma::vec HAMUnit<E, J, S, W>::Attention()
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
  std::cerr << "Attention:\n" << leafAttention;
  return leafAttention;
}

template<typename E, typename J, typename S, typename W>
void HAMUnit<E, J, S, W>::Predict(arma::mat&& input, arma::mat&& output) {
  embed.Predict(input, sequence);
  memory.Initialize(sequence);
  output = arma::zeros(input.n_cols, 1);
  std::cerr << "Sequence length is " << input.n_cols << "\n";
  for (t = 0; t < input.n_cols; ++t)
  {
    std::cerr << "Tick " << t << "\n";
    arma::vec attention = Attention();
    arma::vec input = arma::zeros(memory.Leaf(0).n_elem);
    for (size_t i = 0; i < memorySize; ++i)
    {
      input += attention.at(i) * memory.Leaf(i);
      std::cerr << "After reading " << i+1 << " input is:\n" << input;
    }
    output.at(t, 0) = 0; // TODO?

    std::cerr << "Final input is:" << input << "\n" << "Writing.\n";
    // Write phase.
    for (size_t i = 0; i < memorySize; ++i)
    {
      arma::vec prevMemory = memory.Leaf(i);
      memory.Update(i, input);
      memory.Leaf(i) *= attention.at(i);
      memory.Leaf(i) += (1. - attention.at(i)) * prevMemory;
    }
    std::cerr << "Memory rebuilt.\n";
  }
}

} // namespace augmented
} // namespace ann
} // namespace mlpack

#endif