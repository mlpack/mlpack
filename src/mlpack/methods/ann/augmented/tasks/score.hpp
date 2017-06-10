/**
 * @file score.hpp
 * @author Konstantin Sidorov
 *
 * Definition of scoring functions for sequence prediction problems.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_AUGMENTED_TASKS_SCORE_HPP
#define MLPACK_METHODS_AUGMENTED_TASKS_SCORE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace scorers /* Scoring utilities for augmented */ {

/**
* Function that computes the sequences precision
* (number of correct sequences / number of sequences)
* of model's answer against ground truth answer.
* This version works with sequences of bits.
* 
* @param trueOutputs Ground truth sequences.
* @param predOutputs Sequences predicted by model.
*/
double SequencePrecision(arma::field<arma::colvec> trueOutputs,
                         arma::field<arma::colvec> predOutputs);

/**
* Function that computes the sequences precision
* (number of correct sequences / number of sequences)
* of model's answer against ground truth answer.
* This version works with sequences of binary numbers.
*
* @param trueOutputs Ground truth sequences.
* @param predOutputs Sequences predicted by model.
*/
double SequencePrecision(arma::field<arma::mat> trueOutputs,
                         arma::field<arma::mat> predOutputs);

} // namespace scorers
} // namespace augmented
} // namespace ann
} // namespace mlpack

#include "score_impl.hpp"

#endif
