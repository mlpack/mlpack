/**
 * @file utils.cpp
 * @author _____
 *
 * Implementation of some usefull functions for proprocess the data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "utils.hpp"
#include <stdexcept>

using namespace arma;

void preprocess_data(const mat& data,
		     const rowvec& responses,
		     bool fit_intercept,
		     bool normalize,
		     mat& data_proc,
		     rowvec& responses_proc,
		     colvec& data_offset,
		     colvec& data_scale,
		     double& responses_offset)
{
  // Initialize the offsets to their neutral forms.
  data_offset = zeros<colvec>(data.n_rows);
  data_scale = ones<colvec>(data.n_rows);
  responses_offset = 0.0;

  if (fit_intercept)
    {
      data_offset = mean(data, 1);
      responses_offset = mean(responses);
    }
  if (normalize)
      data_scale = stddev(data, 0, 1);

  // Copy data and response before the processing.
  data_proc = data;
  responses_proc = responses;
  // Center the data.
  data_proc.each_col() -= data_offset;
  // Scale the data.
  data_proc.each_col() /= data_scale;
  // Center the responses.
  responses_proc -= responses_offset;
}
    

