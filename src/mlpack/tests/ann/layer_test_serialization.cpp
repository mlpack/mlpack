/**
 * @file layer_serialization_test.cpp
 * @author Ryan Curtin
 *
 * This file includes the subset of tests in ann/layer/ involving serialization,
 * which is taxing on machines with limited memory such as those used for
 * continuous integration. See file 'layer_test.cpp' for general concerns
 * about compilation demands. While keeping the non-serializing tests in one file
 * reduces compilation time and memory usage, having the serialization tests here
 * permit switching them on or off from cmake.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_ENABLE_ANN_SERIALIZATION
  #define MLPACK_ENABLE_ANN_SERIALIZATION
#endif

#include "layer/batch_norm_serialization.cpp"
