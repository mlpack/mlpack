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
  LinkGene(ssize_t fromNeuronId,
  	       ssize_t toNeuronId,
  	       ssize_t innovationId,
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

  // Set aFromNeuronId.
  void FromNeuronId(ssize_t fromNeuronId) { aFromNeuronId = fromNeuronId; } 

  // Get aFromNeuronId.
  ssize_t FromNeuronId() const { return aFromNeuronId; }

  // Set aToNeuronId.
  void ToNeuronId(ssize_t toNeuronId) { aToNeuronId = toNeuronId; }

  // Get aToNeuronId.
  ssize_t ToNeuronId() const { return aToNeuronId; }

  // Set aInnovationId.
  void InnovationId(ssize_t innovationId) { aInnovationId = innovationId; }

  // Get aInnovationId.
  ssize_t InnovationId() const { return aInnovationId; }

  // Set aWeight.
  void Weight(double weight) { aWeight = weight; }

  // Get aWeight.
  double Weight() const { return aWeight; }

  // Set aEnabled.
  void Enabled(bool enabled) { aEnabled = enabled; }

  // Get aEnabled.
  bool Enabled() const { return aEnabled; }

 private:
  // The IDs of neurons connected by this link.
  ssize_t aFromNeuronId, aToNeuronId;

  // Link weight.
  double aWeight;

  // Link innovation ID.
  ssize_t aInnovationId;

  // Enabled or not.
  bool aEnabled;
 
};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_LINK_GENE_HPP
