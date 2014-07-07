/*
 *  @file: SimpleWeightUpdate.hpp
 *  @author: Udit Saxena
 *
 */

#ifndef _MLPACK_METHOD_PERCEPTRON_LEARN_SIMPLEWEIGHTUPDATE
#define _MLPACK_METHOD_PERCEPTRON_LEARN_SIMPLEWEIGHTUPDATE

#include <mlpack/core.hpp>
/*
This class is used to update the weightVectors matrix according to 
the simple update rule as discussed by Rosenblatt:
  if a vector x has been incorrectly classified by a weight w, 
  then w = w - x
  and  w'= w'+ x
  where w' is the weight vector which correctly classifies x.
*/
namespace mlpack {
namespace perceptron {

class SimpleWeightUpdate 
{
public:
  SimpleWeightUpdate()
  { }
  /*
  This function is called to update the weightVectors matrix. 
  It decreases the weights of the incorrectly classified class while
  increasing the weight of the correct class it should have been classified to.
  
  @param: trainData - the training dataset.
  @param: weightVectors - matrix of weight vectors.
  @param: rowIndex - index of the row which has been incorrectly predicted.
  @param: labelIndex - index of the vector in trainData.
  @param: vectorIndex - index of the class which should have been predicted.
 */
  void UpdateWeights(const arma::mat& trainData, arma::mat& weightVectors,
                     size_t labelIndex, size_t vectorIndex, size_t rowIndex )
  {
    arma::mat instance = trainData.col(labelIndex);
  
    weightVectors.row(rowIndex) = weightVectors.row(rowIndex) - 
                               instance.t();

    weightVectors.row(vectorIndex) = weightVectors.row(vectorIndex) + 
                                 instance.t();
  }
};
}; // namespace perceptron
}; // namespace mlpack

#endif