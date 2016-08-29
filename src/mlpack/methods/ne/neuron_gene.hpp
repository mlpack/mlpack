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
    id(id),
    type(type),
    actFuncType(actFuncType),
    depth(depth),
    coordinate(coordinate),
    input(input),
    activation(activation)
  {}

  /**
   * Copy constructor.
   * 
   * @param neuronGene The neuron to be copied.
   */
  NeuronGene(const NeuronGene& neuronGene)
  {
  	id = neuronGene.id;
  	type = neuronGene.type;
  	actFuncType = neuronGene.actFuncType;
    depth = neuronGene.depth;
    coordinate = neuronGene.coordinate;
  	input = neuronGene.input;
    activation = neuronGene.activation;
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
      id = neuronGene.id;
      type = neuronGene.type;
      actFuncType = neuronGene.actFuncType;
      depth = neuronGene.depth;
      coordinate = neuronGene.coordinate;
      input = neuronGene.input;
      activation = neuronGene.activation;
    }
    
    return *this;
  }  

  /**
   * Get neuron id.
   */
  int Id() const { return id; }

  /**
   * Set neuron id.
   */
  void Id(int id) { this->id = id; }

  /**
   * Get neuron type.
   */
  NeuronType Type() const { return type; }

  /**
   * Set neuron type.
   */
  void Type(NeuronType type) { this->type = type; }

  /**
   * Get activation function type.
   */
  ActivationFuncType ActFuncType() const { return actFuncType; }

  /**
   * Set activation function type.
   */
  void ActFuncType(ActivationFuncType actFuncType) { this->actFuncType = actFuncType; }

  /**
   * Get input.
   */
  double Input() const { return input; }

  /**
   * Set input.
   */
  void Input(double input) { this->input = input; }

  /**
   * Get activation.
   */
  double Activation() const { return activation; }

  /**
   * Set activation.
   */
  void Activation(double activation) { this->activation = activation; }

  /**
   * Get neuron depth.
   */
  double Depth() const { return depth; }

  /**
   * Set neuron depth.
   */
  void Depth(double depth) { this->depth = depth; }

  /**
   * Get neuron coordinates.
   */
  std::vector<double> Coordinate() const { return coordinate; }

  /**
   * Set neuron coordinates.
   */
  void Coordinate(const std::vector<double>& coordinate) { this->coordinate = coordinate; }

  /**
   * Calculate activation based on current input.
   */
  void CalcActivation()
  {
    switch (actFuncType)
    { // TODO: more cases.
      case SIGMOID:                   
        activation = ann::LogisticFunction::fn(input);
        break;
      case TANH:
        activation = ann::TanhFunction::fn(input);
        break;
      case RELU:
        activation = ann::RectifierFunction::fn(input);
        break;
      case LINEAR:
        activation = input;
        break;
      default:
        activation = ann::LogisticFunction::fn(input);
        break;
    }
  }

 private:
  //! Neuron id.
  int id;

  //! Neuron type.
  NeuronType type;

  //! Activation function type.
  ActivationFuncType actFuncType;

  //! Input sum.
  double input;

  //! Activation.
  double activation;

  //! Depth. INPUT and BIAS is 0, OUTPUT is 1. HIDDEN is between 0 and 1. 
  //! Calculate activate by sequence.
  double depth;

  //! Coordinate of neuron.
  std::vector<double> coordinate;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_NEURON_GENE_HPP
