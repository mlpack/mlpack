/**
 * @file link_gene.hpp
 * @author Bang Liu
 *
 * Definition of the LinkGene class.
 */
#ifndef MLPACK_METHODS_NE_LINK_GENE_HPP
#define MLPACK_METHODS_NE_LINK_GENE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ne {

/**
 * This class defines a link gene.
 */
class LinkGene
{
 public:
  /**
   * Default constructor.
   */
  LinkGene() {}

  /**
   * Parametric constructor. Construct the LinkGene object with the given parameters.
   *
   * @param fromNeuronId The id of from neuron.
   * @param toNeuronId The id of to neuron.
   * @param innovationId Innovation id of this link.
   * @param weight Weight of this link.
   * @param enabled Whether this link is enabled or not.
   */
  LinkGene(int fromNeuronId,
  	       int toNeuronId,
  	       int innovationId,
  	       double weight,
           bool enabled):
    aFromNeuronId(fromNeuronId),
    aToNeuronId(toNeuronId),
    aInnovationId(innovationId),
    aWeight(weight),
    aEnabled(enabled)
  {}

  /**
   * Copy constructor. Construct this link by copy another link.
   *
   * @param linkGene The link used to copy.
   */
  LinkGene(const LinkGene& linkGene)
  {
  	aFromNeuronId = linkGene.aFromNeuronId;
  	aToNeuronId = linkGene.aToNeuronId;
  	aInnovationId = linkGene.aInnovationId;
  	aWeight = linkGene.aWeight;
    aEnabled = linkGene.aEnabled;
  }

  /**
   * Destructor.
   */
  ~LinkGene() {}

  /**
   * Operator =.
   *
   * @param linkGene The link to be compared with.
   */
  LinkGene& operator =(const LinkGene& linkGene)
  {
    if (this != &linkGene)
    {
      aFromNeuronId = linkGene.aFromNeuronId;
      aToNeuronId = linkGene.aToNeuronId;
      aInnovationId = linkGene.aInnovationId;
      aWeight = linkGene.aWeight;
      aEnabled = linkGene.aEnabled;
    }
    
    return *this;
  }

  /**
   * Set aFromNeuronId.
   *
   * @param fromNeuronId The value to set.
   */
  void FromNeuronId(int fromNeuronId) { aFromNeuronId = fromNeuronId; } 

  /**
   * Get aFromNeuronId.
   */
  int FromNeuronId() const { return aFromNeuronId; }

  /**
   * Set aToNeuronId.
   *
   * @param toNeuronId The value to set.
   */
  void ToNeuronId(int toNeuronId) { aToNeuronId = toNeuronId; }

  /**
   * Get aToNeuronId.
   */
  int ToNeuronId() const { return aToNeuronId; }

  /**
   * Set aInnovationId.
   *
   * @param innovationId The value to set.
   */
  void InnovationId(int innovationId) { aInnovationId = innovationId; }

  /**
   * Get aInnovationId.
   */
  int InnovationId() const { return aInnovationId; }

  /**
   * Set aWeight.
   *
   * @param weight The value to set.
   */
  void Weight(double weight) { aWeight = weight; }

  /**
   * Get aWeight.
   */
  double Weight() const { return aWeight; }

  /**
   * Set aEnabled.
   *
   * @param enabled The value to set.
   */
  void Enabled(bool enabled) { aEnabled = enabled; }

  /**
   * Get aEnabled.
   */
  bool Enabled() const { return aEnabled; }

 private:
  //! The if of neuron that this link starts from.
  int aFromNeuronId;

  //! The id of neuron that this link ends to.
  int aToNeuronId;

  //! Link weight.
  double aWeight;

  //! Link innovation ID.
  int aInnovationId;

  //! Enabled or not.
  bool aEnabled;
 
};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_LINK_GENE_HPP
