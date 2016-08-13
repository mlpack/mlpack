/**
 * @file neuron_gene.hpp
 * @author Bang Liu
 *
 * Definition of the NeuronGene class.
 */
#ifndef MLPACK_METHODS_NE_NEURON_GENE_HPP
#define MLPACK_METHODS_NE_NEURON_GENE_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>

namespace mlpack {
namespace ne {

/**
 * This is enumeration of neuron types.
 */
enum NeuronType
{
  NONE = 0,
  INPUT,
  BIAS,
  HIDDEN,
  OUTPUT
};

/**
 * This is enumeration of activation function types.
 */
enum ActivationFuncType
{
    SIGMOID = 0,
    TANH,
    LINEAR,
    RELU,

    COUNT  //  Used to count how many activation types.
};

/**
 * This class defines a neuron gene.
 */
class NeuronGene
{
 public:
  /**
   * Default constructor.
   */
  NeuronGene() {}
  
  /**
   * Parametric constructor.
   *
   * @param id Neuron's ID.
   * @param type Neuron's type: INPUT, BIAS, HIDDEN, OUTPUT.
   * @param actFuncType The type of activation function.
   * @param depth The depth of the neuron.
   * @param coordinate The coordinate of the neuron.
   * @param input Neuron's input value.
   * @param activation Neuron's activation value.
   */
  NeuronGene(int id,
  	         NeuronType type,
  	         ActivationFuncType actFuncType,
             double depth,
             std::vector<double> coordinate,
  	         double input,
             double activation):
    aId(id),
    aType(type),
    aActFuncType(actFuncType),
    aDepth(depth),
    aCoordinate(coordinate),
    aInput(input),
    aActivation(activation)
  {}

  /**
   * Copy constructor.
   * 
   * @param neuronGene The neuron to be copied.
   */
  NeuronGene(const NeuronGene& neuronGene)
  {
  	aId = neuronGene.aId;
  	aType = neuronGene.aType;
  	aActFuncType = neuronGene.aActFuncType;
    aDepth = neuronGene.aDepth;
    aCoordinate = neuronGene.aCoordinate;
  	aInput = neuronGene.aInput;
    aActivation = neuronGene.aActivation;
  }

  /**
   * Destructor.
   */
  ~NeuronGene() {}

  /**
   * Operator =.
   *
   * @param neuronGene The neuron to be compared with.
   */
  NeuronGene& operator =(const NeuronGene& neuronGene)
  {
    if (this != &neuronGene)
    {
      aId = neuronGene.aId;
      aType = neuronGene.aType;
      aActFuncType = neuronGene.aActFuncType;
      aDepth = neuronGene.aDepth;
      aCoordinate = neuronGene.aCoordinate;
      aInput = neuronGene.aInput;
      aActivation = neuronGene.aActivation;
    }
    
    return *this;
  }  

  /**
   * Get neuron id.
   */
  int Id() const { return aId; }

  /**
   * Set neuron id.
   */
  void Id(int id) { aId = id; }

  /**
   * Get neuron type.
   */
  NeuronType Type() const { return aType; }

  /**
   * Set neuron type.
   */
  void Type(NeuronType type) { aType = type; }

  /**
   * Get activation function type.
   */
  ActivationFuncType ActFuncType() const { return aActFuncType; }

  /**
   * Set activation function type.
   */
  void ActFuncType(ActivationFuncType actFuncType) { aActFuncType = actFuncType; }

  /**
   * Get input.
   */
  double Input() const { return aInput; }

  /**
   * Set input.
   */
  void Input(double input) { aInput = input; }

  /**
   * Get activation.
   */
  double Activation() const { return aActivation; }

  /**
   * Set activation.
   */
  void Activation(double activation) { aActivation = activation; }

  /**
   * Get neuron depth.
   */
  double Depth() const { return aDepth; }

  /**
   * Set neuron depth.
   */
  void Depth(double depth) { aDepth = depth; }

  /**
   * Get neuron coordinates.
   */
  std::vector<double> Coordinate() const { return aCoordinate; }

  /**
   * Set neuron coordinates.
   */
  void Coordinate(const std::vector<double>& coordinate) { aCoordinate = coordinate; }

  /**
   * Calculate activation based on current input.
   */
  void CalcActivation()
  {
    switch (aActFuncType)
    { // TODO: more cases.
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
  //! Neuron id.
  int aId;

  //! Neuron type.
  NeuronType aType;

  //! Activation function type.
  ActivationFuncType aActFuncType;

  //! Input sum.
  double aInput;

  //! Activation.
  double aActivation;

  //! Depth. INPUT and BIAS is 0, OUTPUT is 1. HIDDEN is between 0 and 1. 
  //! Calculate activate by sequence.
  double aDepth;

  //! Coordinate of neuron.
  std::vector<double> aCoordinate;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_NEURON_GENE_HPP
