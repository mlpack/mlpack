#ifndef MLPACK_METHODS_NEURO_CMAES_LINK_GENE_HPP
#define MLPACK_METHODS_NEURO_CMAES_LINK_GENE_HPP

#include <cstddef>

#include <mlpack/core.hpp>

namespace mlpack {
namespace neuro_cmaes {

/**
 * This class defines a link for the feed forward neural network.
 */
class LinkGene {
 public:
  // Default constructor.
  LinkGene() {}

  // Parametric constructor.
  LinkGene(ssize_t fromNeuronId, ssize_t toNeuronId, double weight):

    aFromNeuronId(fromNeuronId),
    aToNeuronId(toNeuronId),
    aWeight(weight)

  {}

  // Copy constructor.
  LinkGene(const LinkGene& linkGene)
   {
  	aFromNeuronId = linkGene.aFromNeuronId;
  	aToNeuronId = linkGene.aToNeuronId;
  	aWeight = linkGene.aWeight;
  }

  // Destructor.
  ~LinkGene() {}

  // Operator =.
  LinkGene& operator =(const LinkGene& linkGene)
   {
    if (this != &linkGene)
     {
      aFromNeuronId = linkGene.aFromNeuronId;
      aToNeuronId = linkGene.aToNeuronId;
      aWeight = linkGene.aWeight;
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

  // Set aWeight.
  void Weight(double weight) { aWeight = weight; }

  // Get aWeight.
  double Weight() const { return aWeight; }


 private:
  // The IDs of neurons connected by this link.
  ssize_t aFromNeuronId, aToNeuronId;

  // Link weight.
  double aWeight;
};

}  // namespace neuro_cmaes
}  // namespace mlpack

#endif  // MLPACK_METHODS_NEURO_CMAES_LINK_GENE_HPP
