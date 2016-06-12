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
  // Input sum.
  double aInput;

  // Activation.
  double aActivation;

  // Default constructor.
  NeuronGene() {}

  // Parametric constructor.
  NeuronGene(size_t id,
  	         NeuronType type,
  	         ActivationFuncType actFuncType,
  	         double input,
             double activation):
    aId(id),
    aType(type),
    aActFuncType(actFuncType),
    aInput(input),
    aActivation(activation)
  {}

  // Copy constructor.
  NeuronGene(const NeuronGene& neuronGene) {
  	aId = neuronGene.aId;
  	aType = neuronGene.aType;
  	aActFuncType = neuronGene.aActFuncType;
  	aInput = neuronGene.aInput;
    aActivation = neuronGene.aActivation;
  }

  // Destructor.
  ~NeuronGene() {}

  // Get neuron id.
  size_t Id() const { return aId; }

  // Get neuron type.
  NeuronType Type() const { return aType; }

  // Get activation function type.
  ActivationFuncType ActFuncType() const { return aActFuncType; }

  // Set activation function type.
  void ActFuncType(ActivationFuncType actFuncType) {
    aActFuncType = actFuncType;
  }

  // Operator =.
  NeuronGene& operator =(const NeuronGene& neuronGene) {
    if (this != &neuronGene) {
  	  aId = neuronGene.aId;
  	  aType = neuronGene.aType;
  	  aActFuncType = neuronGene.aActFuncType;
  	  aInput = neuronGene.aInput;
      aActivation = neuronGene.aActivation;
    }
    
    return *this;
  }  

 private:
  // Neuron id.
  size_t aId;

  // Neuron type.
  NeuronType aType;

  // Activation function type.
  ActivationFuncType aActFuncType;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_NEURON_GENE_HPP
