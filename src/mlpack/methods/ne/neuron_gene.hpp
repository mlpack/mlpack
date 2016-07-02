/**
 * @file neuron_gene.hpp
 * @author Bang Liu
 *
 * Definition of the NeuronGene class.
 */
#ifndef MLPACK_METHODS_NE_NEURON_GENE_HPP
#define MLPACK_METHODS_NE_NEURON_GENE_HPP

#include <cstddef>

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>

namespace mlpack {
namespace ne {

/**
 * This is enumeration of neuron types.
 */
enum NeuronType {
  NONE = 0,
  INPUT,
  BIAS,
  HIDDEN,
  OUTPUT
};

/**
 * This is enumeration of activation function types.
 */
enum ActivationFuncType {
    SIGMOID = 0,
    TANH,
    LINEAR,
    RELU,
};

/**
 * This class defines a neuron gene.
 */
class NeuronGene {
 public:
  // Default constructor.
  NeuronGene() {}

  // Parametric constructor.
  NeuronGene(ssize_t id,
  	         NeuronType type,
  	         ActivationFuncType actFuncType,
             double depth,
  	         double input,
             double activation):
    aId(id),
    aType(type),
    aActFuncType(actFuncType),
    aDepth(depth),
    aInput(input),
    aActivation(activation)
  {}

  // Copy constructor.
  NeuronGene(const NeuronGene& neuronGene) {
  	aId = neuronGene.aId;
  	aType = neuronGene.aType;
  	aActFuncType = neuronGene.aActFuncType;
    aDepth = neuronGene.aDepth;
  	aInput = neuronGene.aInput;
    aActivation = neuronGene.aActivation;
  }

  // Destructor.
  ~NeuronGene() {}

  // Operator =.
  NeuronGene& operator =(const NeuronGene& neuronGene) {
    if (this != &neuronGene) {
      aId = neuronGene.aId;
      aType = neuronGene.aType;
      aActFuncType = neuronGene.aActFuncType;
      aDepth = neuronGene.aDepth;
      aInput = neuronGene.aInput;
      aActivation = neuronGene.aActivation;
    }
    
    return *this;
  }  

  // Get neuron id.
  ssize_t Id() const { return aId; }

  // Set neuron id.
  void Id(ssize_t id) { aId = id; }

  // Get neuron type.
  NeuronType Type() const { return aType; }

  // Set neuron type.
  void Type(NeuronType type) { aType = type; }

  // Get activation function type.
  ActivationFuncType ActFuncType() const { return aActFuncType; }

  // Set activation function type.
  void ActFuncType(ActivationFuncType actFuncType) {
    aActFuncType = actFuncType;
  }

  // Get input.
  double Input() const { return aInput; }

  // Set input.
  void Input(double input) { aInput = input; }

  // Get activation.
  double Activation() const { return aActivation; }

  // Set activation.
  void Activation(double activation) { aActivation = activation; }

  // Get neuron depth.
  double Depth() const { return aDepth; }

  // Set neuron depth.
  void Depth(double depth) { aDepth = depth; }

  // Calculate activation based on current input.
  void CalcActivation() {
    switch (aType) { // TODO: more cases.
      case SIGMOID:                   
        aActivation = ann::LogisticFunction::fn(aInput);
        break;
      case TANH:
        aActivation = ann::TanhFunction::fn(aInput);
        break;
      case RELU:
        aActivation = ann::RectifierFunction::fn(aInput);
        break;
      case LINEAR:
        aActivation = aInput;
        break;
      default:
        aActivation = ann::LogisticFunction::fn(aInput);
        break;
    }
  }

 private:
  // Neuron id.
  ssize_t aId;

  // Neuron type.
  NeuronType aType;

  // Activation function type.
  ActivationFuncType aActFuncType;

  // Input sum.
  double aInput;

  // Activation.
  double aActivation;

  // Depth. INPUT and BIAS is 0, OUTPUT is 1. HIDDEN is between 0 and 1. Calculate activate by sequence.
  double aDepth;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_NEURON_GENE_HPP
