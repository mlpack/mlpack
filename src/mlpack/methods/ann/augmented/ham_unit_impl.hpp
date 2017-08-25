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
    t(0), reset(false)
{
  // Nothing to do here.
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
  for (size_t node = 1; node < nodesCnt; ++node)
  {
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
void HAMUnit<E, J, S, W, C>::Forward(arma::mat&& input, arma::mat&& output)
{
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
void HAMUnit<E, J, S, W, C>::ResetParameters()
{
  embed.ResetParameters();
  search.ResetParameters();
  controller.ResetParameters();
  memory.ResetParameters();

  const size_t embedCount = embed.Parameters().n_elem;
  const size_t searchCount = search.Parameters().n_elem;
  const size_t controllerCount = controller.Parameters().n_elem;
  const size_t joinCount = memory.JoinObject().Parameters().n_elem;
  const size_t writeCount = memory.WriteObject().Parameters().n_elem;

  parameters = arma::mat(embedCount + searchCount + controllerCount +
      joinCount + writeCount, 1);

  NetworkInitialization<> networkInit;
  networkInit.Initialize(embed.Model(), parameters);
  networkInit.Initialize(search.Model(), parameters, embedCount);
  networkInit.Initialize(controller.Model(), parameters, embedCount +
      searchCount);
  networkInit.Initialize(memory.JoinObject().Model(), parameters, embedCount +
      searchCount + controllerCount);
  networkInit.Initialize(memory.WriteObject().Model(), parameters, embedCount +
      searchCount + controllerCount + joinCount);

  reset = true;
}
} // namespace augmented
} // namespace ann
} // namespace mlpack

#endif
