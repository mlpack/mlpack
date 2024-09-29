/**
 * @file methods/ann/loss_functions/loss_functions.hpp
 * @author Ryan Curtin
 *
 * Convenience include for all loss functions implemented for mlpack's neural
 * network toolkit.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_LOSS_FUNCTIONS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_LOSS_FUNCTIONS_HPP

#include "binary_cross_entropy_loss.hpp"
#include "cosine_embedding_loss.hpp"
#include "dice_loss.hpp"
#include "earth_mover_distance.hpp"
#include "empty_loss.hpp"
#include "hinge_embedding_loss.hpp"
#include "hinge_loss.hpp"
#include "huber_loss.hpp"
#include "kl_divergence.hpp"
#include "l1_loss.hpp"
#include "log_cosh_loss.hpp"
#include "margin_ranking_loss.hpp"
#include "mean_absolute_percentage_error.hpp"
#include "mean_bias_error.hpp"
#include "mean_squared_error.hpp"
#include "mean_squared_logarithmic_error.hpp"
#include "multilabel_softmargin_loss.hpp"
#include "negative_log_likelihood.hpp"
#include "poisson_nll_loss.hpp"
#include "reconstruction_loss.hpp"
#include "sigmoid_cross_entropy_error.hpp"
#include "soft_margin_loss.hpp"
#include "triplet_margin_loss.hpp"
#include "vr_class_reward.hpp"

#endif
