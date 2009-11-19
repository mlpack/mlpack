/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
#ifndef RIDGE_REGRESSION_UTIL_H
#define RIDGE_REGRESSION_UTIL_H

#include "ridge_regression.h"

class RidgeRegressionUtil {

 public:

  template<typename T>
  static void CopyVectorExceptOneIndex_(const GenVector<T> &source,
					index_t exclude_index,
					GenVector<T> *destination) {
    destination->Init(source.length() - 1);
    index_t current_index = 0;

    for(index_t j = 0; j < source.length(); j++) {
      if(source[j] != exclude_index) {
	(*destination)[current_index] = source[j];
	current_index++;
      }
    }
  }

  static double SquaredCorrelationCoefficient(const Vector &observations,
					      const Vector &predictions) {
    
    // Compute the average of the observed values.
    double avg_observed_value = 0;
    
    for(index_t i = 0; i < observations.length(); i++) {
      avg_observed_value += observations[i];
    }
    avg_observed_value /= ((double) observations.length());

    // Compute something proportional to the variance of the observed
    // values, and the sum of squared residuals of the predictions
    // against the observations.
    double variance = 0;
    double residual = 0;
    for(index_t i = 0; i < observations.length(); i++) {
      variance += math::Sqr(observations[i] - avg_observed_value);
      residual += math::Sqr(observations[i] - predictions[i]);
    }
    return (variance - residual) / variance;
  }

  static double VarianceInflationFactor(const Vector &observations,
					const Vector &predictions) {
    
    return 1.0 / 
      (1.0 - SquaredCorrelationCoefficient(observations, predictions));
  }

};

#endif
