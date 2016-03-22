/**
 * @file empty_layer.hpp
 * @author Palash Ahuja
 *
 * Definition of the EmptyLayer class, which is basically empty.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_EMPTY_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_EMPTY_LAYER_HPP

namespace mlpack{
namespace ann /** Artificial Neural Network. */ {
/**
 * Definition of an empty layer class which does absolutely nothing.
 */
class EmptyLayer
{
  public:
  /**
   * Creates the empty layer object. All the methods are
   * empty as well.
   */
  EmptyLayer()
  {
    // nothing to do here.
  }
  template<typename eT>
  void Forward(const arma::Mat<eT>&, arma::Mat<eT>&)
  {
    // nothing to do here.
  }

  template<typename InputType, typename eT>
  void Backward(const InputType&,/* unused */
                const arma::Mat<eT>&,
		arma::Mat<eT>&)
  {
    // nothing to do here.
  }

  template<typename eT, typename GradientDataType>
  void Gradient(const arma::Mat<eT>&, GradientDataType&)
  {
    // nothing to do here.
  }

  //! Get the weights.
  arma::mat const& Weights() const { return random; }
  
  //! Modify the weights.
  arma::mat& Weights() { return random; }
  
  //! Get the input parameter.
  arma::mat const& InputParameter() const { return random; }
  
  //! Modify the input parameter.
  arma::mat& InputParameter() { return random; }

  //! Get the output parameter.
  arma::mat const& OutputParameter() const { return random; }

  //! Modify the output parameter.
  arma::mat& OutputParameter() { return random; }

  //! Get the delta.
  arma::mat const& Delta() const { return random; }
  
  //! Modify the delta.
  arma::mat& Delta() { return random; }

  //! Get the gradient.
  arma::mat const& Gradient() const { return random; }

  //! Modify the gradient.
  arma::mat& Gradient() { return random; } 
  
  //! something random.
  arma::mat random;

}; // class EmptyLayer

} //namespace ann
} //namespace mlpack

#endif
