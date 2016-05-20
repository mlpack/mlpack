/**
 * @file gene.hpp
 * @author Bang Liu
 *
 * Definition of the LinkGene and NeuronGene class.
 */
#ifndef MLPACK_METHODS_NE_GENE_HPP
#define MLPACK_METHODS_NE_GENE_HPP

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
 * This class defines a link gene.
 */
class LinkGene {
 public:
  // Default constructor.
  LinkGene() {}

  // Parametric constructor.
  LinkGene(unsigned int fromNeuronId,
  	       unsigned int toNeuronId,
  	       unsigned int innovationId,
  	       double weight):
    aFromNeuronId(fromNeuronId),
    aToNeuronId(toNeuronId),
    aInnovationId(innovationId),
    aWeight(weight)
  {}

  // Copy constructor.
  LinkGene(const LinkGene& linkGene) {
  	aFromNeuronId = linkGene.aFromNeuronId;
  	aToNeuronId = linkGene.aToNeuronId;
  	aInnovationId = linkGene.aInnovationId;
  	aWeight = linkGene.aWeight;
  }

  // Destructor.
  ~LinkGene() {}

  // Get aFromNeuronId.
  unsigned int FromNeuronId() const { return aFromNeuronId; }

  // Get aToNeuronId.
  unsigned int ToNeuronId() const { return aToNeuronId; }

  // Get aInnovationId.
  unsigned int InnovationId() const { return aInnovationId; }

  // Set aWeight.
  void Weight(double weight) { aWeight = weight; }

  // Get aWeight.
  double Weight() const { return aWeight; }

  // Operator =.
  LinkGene& operator =(const LinkGene& linkGene) {
    if (this != &linkGene) {
  	  aFromNeuronId = linkGene.aFromNeuronId;
  	  aToNeuronId = linkGene.aToNeuronId;
  	  aInnovationId = linkGene.aInnovationId;
  	  aWeight = linkGene.aWeight;
    }
    
    return *this;
  }

 private:
  // The IDs of neurons connected by this link.
  unsigned int aFromNeuronId, aToNeuronId;

  // Link weight.
  double aWeight;

  // Link innovation ID.
  unsigned int aInnovationId;
 
};

/**
 * This class defines a neuron gene.
 */
class NeuronGene {
 public:
  // Default constructor.
  NeuronGene() {}

  // Parametric constructor.
  NeuronGene(int id,
  	         NeuronType type,
  	         ActivationFuncType actFuncType):
    aId(id),
    aType(type),
    aActFuncType(actFuncType)
  {}

  // Copy constructor.
  NeuronGene(NeuronGene& neuronGene) {
  	aId = neuronGene.aId;
  	aType = neuronGene.aType;
  	aActFuncType = neuronGene.aActFuncType;
  }

  // Destructor.
  ~NeuronGene() {}

  // Get neuron id.
  unsigned int Id() const { return aId; }

  // Get neuron type.
  NeuronType Type() const { return aType; }

  // Get activation function type.
  ActivationFuncType ActFuncType() const { return aActFuncType; }

 private:
  // Neuron id.
  unsigned int aId;

  // Neuron type.
  NeuronType aType;

  // Activation function type.
  ActivationFuncType aActFuncType;

  // Operator =.
  NeuronGene& operator =(const NeuronGene& neuronGene) {
    if (this != &neuronGene) {
  	  aId = neuronGene.aId;
  	  aType = neuronGene.aType;
  	  aActFuncType = neuronGene.aActFuncType;
    }
    
    return *this;
  }  

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_GENE_HPP
