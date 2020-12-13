/**
 * @file check_input_shape.hpp
 * @author Khizir Siddiqui
 * @author Nippun Sharma
 * 
 * Definition of the CheckInputShape() function that checks
 * whether the shape of input is consistent with the first layer
 * of the neural network.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_CHECK_INPUT_SHAPE_HPP
#define MLPACK_METHODS_ANN_UTIL_CHECK_INPUT_SHAPE_HPP

#include <mlpack/methods/ann/visitor/input_shape_visitor.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

template<typename T>
void CheckInputShape(const T& network, const size_t inputShape, 
                     const std::string& functionName)
{
  for (size_t l=0; l<network.size(); ++l)
  {
    size_t layerInShape = boost::apply_visitor(InShapeVisitor(), network[l]);
    if (layerInShape == 0)
    {
      continue;
    }
    else if (layerInShape == inputShape)
    {
      break;
    }
    else
    {
      std::string estr = functionName + ": "; 
                  estr += "the first layer of the network expects ";
                  estr += std::to_string(layerInShape) + " elements, ";
                  estr += "but the input has " + std::to_string(inputShape) + " dimensions! ";
      throw std::logic_error(estr);
    }
  }
}

} // namespace ann
} // namespace mlpack

#endif