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
  size_t FromNeuronId() const { return aFromNeuronId; }

  // Get aToNeuronId.
  size_t ToNeuronId() const { return aToNeuronId; }

  // Get aInnovationId.
  size_t InnovationId() const { return aInnovationId; }

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
  size_t aFromNeuronId, aToNeuronId;

  // Link weight.
  double aWeight;

  // Link innovation ID.
  size_t aInnovationId;
 
};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_GENE_HPP
