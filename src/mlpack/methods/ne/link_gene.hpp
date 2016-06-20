/**
 * @file link_gene.hpp
 * @author Bang Liu
 *
 * Definition of the LinkGene class.
 */
#ifndef MLPACK_METHODS_NE_LINK_GENE_HPP
#define MLPACK_METHODS_NE_LINK_GENE_HPP

#include <cstddef>

#include <mlpack/core.hpp>

namespace mlpack {
namespace ne {

/**
 * This class defines a link gene.
 */
class LinkGene {
 public:
  // Default constructor.
  LinkGene() {}

  // Parametric constructor.
  LinkGene(size_t fromNeuronId,
  	       size_t toNeuronId,
  	       size_t innovationId,
  	       double weight,
           bool enabled):
    aFromNeuronId(fromNeuronId),
    aToNeuronId(toNeuronId),
    aInnovationId(innovationId),
    aWeight(weight),
    aEnabled(enabled)
  {}

  // Copy constructor.
  LinkGene(const LinkGene& linkGene) {
  	aFromNeuronId = linkGene.aFromNeuronId;
  	aToNeuronId = linkGene.aToNeuronId;
  	aInnovationId = linkGene.aInnovationId;
  	aWeight = linkGene.aWeight;
    aEnabled = linkGene.aEnabled;
  }

  // Destructor.
  ~LinkGene() {}

  // Set aFromNeuronId.
  void FromNeuronId(size_t fromNeuronId) { aFromNeuronId = fromNeuronId; } 

  // Get aFromNeuronId.
  size_t FromNeuronId() const { return aFromNeuronId; }

  // Set aToNeuronId.
  void ToNeuronId(size_t toNeuronId) { aToNeuronId = toNeuronId; }

  // Get aToNeuronId.
  size_t ToNeuronId() const { return aToNeuronId; }

  // Set aWeight.
  void Weight(double weight) { aWeight = weight; }

  // Get aWeight.
  double Weight() const { return aWeight; }

  // Set aInnovationId.
  void InnovationId(size_t innovationId) { aInnovationId = innovationId; }

  // Get aInnovationId.
  size_t InnovationId() const { return aInnovationId; }

  // Set aEnabled.
  void Enabled(bool enabled) { aEnabled = enabled; }

  // Get aEnabled.
  bool Enabled() const { return aEnabled; }

  // Operator =.
  LinkGene& operator =(const LinkGene& linkGene) {
    if (this != &linkGene) {
  	  aFromNeuronId = linkGene.aFromNeuronId;
  	  aToNeuronId = linkGene.aToNeuronId;
  	  aInnovationId = linkGene.aInnovationId;
  	  aWeight = linkGene.aWeight;
      aEnabled = linkGene.aEnabled;
    }
    
    return *this;
  }

 private:
  // The IDs of neurons connected by this link.
  size_t aFromNeuronId, aToNeuronId;

  // Link weight.
  double aWeight;

  // Link innovation ID.
  size_t aInnovationId;

  // Enabled or not.
  bool aEnabled;
 
};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_LINK_GENE_HPP
