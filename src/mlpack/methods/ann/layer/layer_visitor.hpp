/**
 * @file layer_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an easy way to serialize a layer, abstracts away the
 * different types of layers, and also automatically directs any function to the
 * right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LAYER_VISITOR_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYER_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * LoadOutputParameterVisitor restores the output parameter using the given
 * parameter set.
 */
class LoadOutputParameterVisitor : public boost::static_visitor<void>
{
 public:
  //! Restore the output parameter given a parameter set.
  LoadOutputParameterVisitor(std::vector<arma::mat>&& parameter);

  //! Restore the output parameter.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The parameter set.
  std::vector<arma::mat>&& parameter;

  //! Restore the output parameter for a module which doesn't implement the
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  OutputParameter(T* layer) const;

  //! Restore the output parameter for a module which implements the Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  OutputParameter(T* layer) const;
};

/**
 * SaveOutputParameterVisitor saves the output parameter into the given
 * parameter set.
 */
class SaveOutputParameterVisitor : public boost::static_visitor<void>
{
 public:
  //! Save the output parameter into the given parameter set.
  SaveOutputParameterVisitor(std::vector<arma::mat>&& parameter);

  //! Save the output parameter.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The parameter set.
  std::vector<arma::mat>&& parameter;

  //! Save the output parameter for a module which doesn't implement the
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  OutputParameter(T* layer) const;

  //! Save the output parameter for a module which implements the Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  OutputParameter(T* layer) const;
};

/**
 * DeleteVisitor executes the destructor of the instantiated object.
 */
class DeleteVisitor : public boost::static_visitor<void>
{
 public:
  //! Execute the destructor.
  template<typename LayerType>
  void operator()(LayerType* layer) const;
};

/**
 * ForwardOutputVisitor executes the Forward() function given the input and
 * output parameter.
 */
class ForwardOutputVisitor : public boost::static_visitor<double>
{
 public:
  //! Execute the Foward() function given the input and output parameter.
  ForwardOutputVisitor(arma::mat&& input, arma::mat&& output);

  //! Execute the Foward() function.
  template<typename LayerType>
  double operator()(LayerType* layer) const;

 private:
  //! The input parameter set.
  arma::mat&& input;

  //! The output parameter set.
  arma::mat&& output;
};

/**
 * ForwardVisitor executes the Forward() function given the input and output
 * parameter.
 */
class ForwardVisitor : public boost::static_visitor<void>
{
 public:
  //! Execute the Foward() function given the input and output parameter.
  ForwardVisitor(arma::mat&& input, arma::mat&& output);

  //! Execute the Foward() function.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The input parameter set.
  arma::mat&& input;

  //! The output parameter set.
  arma::mat&& output;
};

/**
 * BackwardVisitor executes the Backward() function given the input, error and
 * delta parameter.
 */
class BackwardVisitor : public boost::static_visitor<void>
{
 public:
  //! Execute the Backward() function given the input, error and delta
  //! parameter.
  BackwardVisitor(arma::mat&& input, arma::mat&& error, arma::mat&& delta);

  //! Execute the Backward() function.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The input parameter set.
  arma::mat&& input;

  //! The error parameter.
  arma::mat&& error;

  //! The delta parameter.
  arma::mat&& delta;
};

/**
 * ResetVisitor executes the Reset() function.
 */
class ResetVisitor : public boost::static_visitor<void>
{
 public:
  //! Execute the Reset() function.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! Execute the Reset() function for a module which implements the Reset()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasResetCheck<T, void(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  ResetParameter(T* layer) const;

  //! Execute the Reset() function for a module which implements the Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasResetCheck<T, void(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  ResetParameter(T* layer) const;

  //! Execute the Reset() function for a module which implements the Reset()
  //! and Model() function.
  template<typename T>
  typename std::enable_if<
      HasResetCheck<T, void(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  ResetParameter(T* layer) const;

  //! Do not execute the Reset() function for a module which doesn't implement
  // the Reset() or Model() function.
  template<typename T>
  typename std::enable_if<
      !HasResetCheck<T, void(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  ResetParameter(T* layer) const;
};

/**
 * InputParameterVisitor exposes the input parameter of the given module.
 */
class InputParameterVisitor : public boost::static_visitor<arma::mat&>
{
 public:
  //! Return the input parameter set.
  template<typename LayerType>
  arma::mat& operator()(LayerType* layer) const;
};

/**
 * OutputParameterVisitor exposes the output parameter of the given module.
 */
class OutputParameterVisitor : public boost::static_visitor<arma::mat&>
{
 public:
  //! Return the output parameter set.
  template<typename LayerType>
  arma::mat& operator()(LayerType* layer) const;
};

/**
 * DeltaVisitor exposes the delta parameter of the given module.
 */
class DeltaVisitor : public boost::static_visitor<arma::mat&>
{
 public:
  //! Return the delta parameter.
  template<typename LayerType>
  arma::mat& operator()(LayerType* layer) const;
};

/**
 * ParametersVisitor exposes the parameters set of the given module and stores
 * the parameters set into the given matrix.
 */
class ParametersVisitor : public boost::static_visitor<void>
{
 public:
  //! Store the parameters set into the given parameters matrix.
  ParametersVisitor(arma::mat&& parameters);

  //! Set the parameters set.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The parameters set.
  arma::mat&& parameters;

  //! Do not set the parameters set if the module doesn't implement the
  //! Parameters() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value, void>::type
  LayerParameters(T* layer, P& output) const;

  //! Set the parameters set if the module implements the Parameters() function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value, void>::type
  LayerParameters(T* layer, P& output) const;
};

/**
 * ParametersSetVisitor update the parameters set using the given matrix.
 */
class ParametersSetVisitor : public boost::static_visitor<void>
{
 public:
  //! Update the parameters set given the parameters matrix.
  ParametersSetVisitor(arma::mat&& parameters);

  //! Update the parameters set.
  template<typename LayerType>
  void operator()(LayerType *layer) const;

 private:
  //! The parameters set.
  arma::mat&& parameters;

  //! Do not update the parameters set if the module doesn't implement the
  //! Parameters() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value, void>::type
  LayerParameters(T* layer, P& output) const;

  //! Update the parameters set if the module implements the Parameters()
  //! function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value, void>::type
  LayerParameters(T* layer, P& output) const;
};

/**
 * WeightSizeVisitor returns the number of weights of the given module.
 */
class WeightSizeVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Return the number of weights.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! If the module doesn't implement the Parameters() or Model() function
  //! return 0.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P& output) const;

  //! Return the number of parameters if the module implements the Model()
  //! function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P& output) const;

  //! Return the number of parameters if the module implements the Parameters()
  //! function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P& output) const;

  //! Return the accumulated number of parameters if the module implements the
  //! Parameters() and Model() function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P& output) const;
};

/**
 * SetInputWidthVisitor updates the input width parameter with the given input
 * width.
 */
class SetInputWidthVisitor : public boost::static_visitor<bool>
{
 public:
  //! Update the input width parameter with the given input width.
  SetInputWidthVisitor(const size_t inputWidth = 0, const bool reset = false);

  //! Update the input width parameter.
  template<typename LayerType>
  bool operator()(LayerType* layer) const;

 private:
  //! The input width parameter.
  size_t inputWidth;

  //! If set reset the height parameter if already set.
  bool reset;

  //! Do nothing if the module doesn't implement the InputWidth() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasInputWidth<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputWidth(T* layer) const;

  //! Update the input width if the module implements the InputWidth() function.
  template<typename T>
  typename std::enable_if<
      HasInputWidth<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputWidth(T* layer) const;

  //! Update the input width if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasInputWidth<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputWidth(T* layer) const;

  //! Update the input width if the module implements the InputWidth() or
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      HasInputWidth<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputWidth(T* layer) const;
};

/**
 * SetInputHeightVisitor updates the input height parameter with the given input
 * height.
 */
class SetInputHeightVisitor : public boost::static_visitor<bool>
{
 public:
  //! Update the input height parameter with the given input height.
  SetInputHeightVisitor(const size_t inputHeight = 0, const bool reset = false);

  //! Update the input height parameter.
  template<typename LayerType>
  bool operator()(LayerType* layer) const;

 private:
  //! The input height parameter.
  size_t inputHeight;

  //! If set reset the height parameter if already set.
  bool reset;

  //! Do nothing if the module doesn't implement the InputHeight() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasInputHeight<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputHeight(T* layer) const;

  //! Update the input height if the module implements the InputHeight()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasInputHeight<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputHeight(T* layer) const;

  //! Update the input height if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasInputHeight<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputHeight(T* layer) const;

  //! Update the input height if the module implements the InputHeight() or
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      HasInputHeight<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputHeight(T* layer) const;
};

/**
 * OutputWidthVisitor exposes the OutputWidth() method of the given module.
 */
class OutputWidthVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Return the output width.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! Return 0 if the module doesn't implement the InputWidth() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasInputWidth<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputWidth(T* layer) const;

  //! Return the output width if the module implements the InputWidth()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasInputWidth<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputWidth(T* layer) const;

  //! Return the output width if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasInputWidth<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputWidth(T* layer) const;

  //! Return the output width if the module implements the Model() or
  //! InputWidth() function.
  template<typename T>
  typename std::enable_if<
      HasInputWidth<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputWidth(T* layer) const;
};

/**
 * OutputWidthVisitor exposes the OutputHeight() method of the given module.
 */
class OutputHeightVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Return the output height.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! Return 0 if the module doesn't implement the InputHeight() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasInputHeight<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;

  //! Return the output height if the module implements the InputHeight()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasInputHeight<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;

  //! Return the output height if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasInputHeight<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;

  //! Return the output height if the module implement the Model() or
  //! InputHeight() function.
  template<typename T>
  typename std::enable_if<
      HasInputHeight<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;
};

/**
 * LastOutputWidthVisitor exposes the OutputWidth() method of the given module.
 */
class LastOutputWidthVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Return the output width.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! Return 0 if the module doesn't implement the InputWidth() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasInputWidth<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputWidth(T* layer) const;

  //! Return the output width if the module implements the InputWidth()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasInputWidth<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputWidth(T* layer) const;

  //! Return the output width if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasInputWidth<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputWidth(T* layer) const;

  //! Return the output width if the module implements the Model() or
  //! InputWidth() function.
  template<typename T>
  typename std::enable_if<
      HasInputWidth<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputWidth(T* layer) const;
};

/**
 * LastOutputHeightVisitor exposes the OutputHeight() method of the given module.
 */
class LastOutputHeightVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Return the output height.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! Return 0 if the module doesn't implement the InputHeight() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasInputHeight<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;

  //! Return the output height if the module implements the InputHeight()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasInputHeight<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;

  //! Return the output height if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasInputHeight<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;

  //! Return the output height if the module implement the Model() or
  //! InputHeight() function.
  template<typename T>
  typename std::enable_if<
      HasInputHeight<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;
};

/**
 * WeightSetVisitor update the module parameters given the parameters set.
 */
class WeightSetVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Update the parameters given the parameters set and offset.
  WeightSetVisitor(arma::mat&& weight, const size_t offset = 0);

  //! Update the parameters set.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! The parameters set.
  arma::mat&& weight;

  //! The parameters offset.
  const size_t offset;

  //! Do not update the parameters if the module doesn't implement the
  //! Parameters() or Model() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P&& input) const;

  //! Update the parameters if the module implements the Model() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P&& input) const;

  //! Update the parameters if the module implements the Parameters() function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P&& input) const;

  //! Update the parameters if the module implements the Model() and
  //! Parameters() function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P&& input) const;
};

/**
 * RhoVisitor exposes the Rho() method of the given module.
 */
class RhoVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Return the output height.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! Return 0 if the module doesn't implement the InputHeight() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasRho<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerRho(T* layer) const;

  //! Return the output height if the module implements the InputHeight()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasRho<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerRho(T* layer) const;

  //! Return the output height if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasRho<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerRho(T* layer) const;

  //! Return the output height if the module implement the Model() or
  //! InputHeight() function.
  template<typename T>
  typename std::enable_if<
      HasRho<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerRho(T* layer) const;
};

/**
 * DeterministicSetVisitor set the deterministic parameter given the
 * deterministic value.
 */
class DeterministicSetVisitor : public boost::static_visitor<void>
{
 public:
  //! Set the deterministic parameter given the current deterministic value.
  DeterministicSetVisitor(const bool deterministic = true);

  //! Set the deterministic parameter.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The deterministic parameter.
  const bool deterministic;

  //! Set the deterministic parameter if the module implements the
  //! Deterministic() and Model() function.
  template<typename T>
  typename std::enable_if<
      HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerDeterministic(T* layer) const;

  //! Set the deterministic parameter if the module implements the
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      !HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerDeterministic(T* layer) const;

  //! Set the deterministic parameter if the module implements the
  //! Deterministic() function.
  template<typename T>
  typename std::enable_if<
      HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerDeterministic(T* layer) const;

  //! Do not set the deterministic parameter if the module doesn't implement the
  //! Deterministic() or Model() function.
  template<typename T>
  typename std::enable_if<
      !HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerDeterministic(T* layer) const;
};

/**
 * AddVisitor exposes the Add() method of the given module.
 */
class AddVisitor : public boost::static_visitor<void>
{
 public:
  //! Exposes the Add() method of the given module.
  template<typename T>
  AddVisitor(T newLayer);

  //! Exposes the Add() method.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The layer that should be added.
  LayerTypes newLayer;

  //! Only add the layer if the module implements the Add() function.
  template<typename T>
  typename std::enable_if<
      HasAddCheck<T, void(T::*)(LayerTypes)>::value, void>::type
  LayerAdd(T* layer) const;

  //! Do not add the layer if the module doesn't implement the Add() function.
  template<typename T>
  typename std::enable_if<
      !HasAddCheck<T, void(T::*)(LayerTypes)>::value, void>::type
  LayerAdd(T* layer) const;
};

/**
 * GradientSetVisitor update the gradient parameter given the gradient set.
 */
class GradientSetVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Update the gradient parameter given the gradient set.
  GradientSetVisitor(arma::mat&& gradient, size_t offset = 0);

  //! Update the gradient parameter.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! The gradient set.
  arma::mat&& gradient;

  //! The gradient offset.
  size_t offset;

  //! Update the gradient if the module implements the Gradient() function.
  template<typename T>
  typename std::enable_if<
      HasGradientCheck<T, arma::mat&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Update the gradient if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasGradientCheck<T, arma::mat&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Update the gradient if the module implements the Gradient() and Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasGradientCheck<T, arma::mat&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Do not update the gradient parameter if the module doesn't implement the
  //! Gradient() or Model() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasGradientCheck<T, P&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, P& input) const;
};


/**
 * GradientUpdateVisitor update the gradient parameter given the gradient set.
 */
class GradientUpdateVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Update the gradient parameter given the gradient set.
  GradientUpdateVisitor(arma::mat&& gradient, size_t offset = 0);

  //! Update the gradient parameter.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! The gradient set.
  arma::mat&& gradient;

  //! The gradient offset.
  size_t offset;

  //! Update the gradient if the module implements the Gradient() function.
  template<typename T>
  typename std::enable_if<
      HasGradientCheck<T, arma::mat&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Update the gradient if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasGradientCheck<T, arma::mat&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Update the gradient if the module implements the Gradient() and Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasGradientCheck<T, arma::mat&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Do not update the gradient parameter if the module doesn't implement the
  //! Gradient() or Model() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasGradientCheck<T, P&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, P& input) const;
};

/*
 * GradientZeroVisitor set the gradient to zero for the given module.
 */
class GradientZeroVisitor : public boost::static_visitor<void>
{
 public:
  //! Set the gradient to zero for the given module.
  GradientZeroVisitor();

  //! Set the gradient to zero.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! Set the gradient to zero if the module implements the Gradient() function.
  template<typename T>
  typename std::enable_if<
      HasGradientCheck<T, arma::mat&(T::*)()>::value, void>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Do not set the gradient to zero if the module doesn't implement the
  //! Gradient() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasGradientCheck<T, P&(T::*)()>::value, void>::type
  LayerGradients(T* layer, P& input) const;
};

/**
 * SearchModeVisitor executes the Gradient() method of the given module using
 * the input and delta parameter.
 */
class GradientVisitor : public boost::static_visitor<void>
{
 public:
  //! Executes the Gradient() method of the given module using the input and
  //! delta parameter.
  GradientVisitor(arma::mat&& input, arma::mat&& delta);

  //! Executes the Gradient() method.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The input set.
  arma::mat&& input;

  //! The delta parameter.
  arma::mat&& delta;

  //! Execute the Gradient() function if the module implements the Gradient()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasGradientCheck<T, arma::mat&(T::*)()>::value, void>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Do not execute the Gradient() function if the module doesn't implement
  //! the Gradient() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasGradientCheck<T, P&(T::*)()>::value, void>::type
  LayerGradients(T* layer, P& input) const;
};

/**
 * RewardSetVisitor set the reward parameter given the reward value.
 */
class RewardSetVisitor : public boost::static_visitor<void>
{
 public:
  //! Set the reward parameter given the reward value.
  RewardSetVisitor(const double reward);

  //! Set the reward parameter.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The reward value.
  const double reward;

  //! Set the deterministic parameter if the module implements the
  //! Deterministic() and Model() function.
  template<typename T>
  typename std::enable_if<
      HasRewardCheck<T, double&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerReward(T* layer) const;

  //! Set the deterministic parameter if the module implements the
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      !HasRewardCheck<T, double&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerReward(T* layer) const;

  //! Set the deterministic parameter if the module implements the
  //! Deterministic() function.
  template<typename T>
  typename std::enable_if<
      HasRewardCheck<T, double&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerReward(T* layer) const;

  //! Do not set the deterministic parameter if the module doesn't implement the
  //! Deterministic() or Model() function.
  template<typename T>
  typename std::enable_if<
      !HasRewardCheck<T, double&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerReward(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "layer_visitor_impl.hpp"

#endif
