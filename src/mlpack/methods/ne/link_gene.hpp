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
    fromNeuronId(fromNeuronId),
    toNeuronId(toNeuronId),
    innovationId(innovationId),
    weight(weight),
    enabled(enabled)
  {}

  // /**
  //  * Copy constructor. Construct this link by copy another link.
  //  *
  //  * @param linkGene The link used to copy.
  //  */
  // LinkGene(const LinkGene& linkGene)
  // {
  // 	fromNeuronId = linkGene.fromNeuronId;
  // 	toNeuronId = linkGene.toNeuronId;
  // 	innovationId = linkGene.innovationId;
  // 	weight = linkGene.weight;
  //   enabled = linkGene.enabled;
  // }

  // /**
  //  * Destructor.
  //  */
  // ~LinkGene() {}

  /**
   * Operator =.
   *
   * @param linkGene The link to be compared with.
   */
  LinkGene& operator =(const LinkGene& linkGene)
  {
    if (this != &linkGene)
    {
      fromNeuronId = linkGene.fromNeuronId;
      toNeuronId = linkGene.toNeuronId;
      innovationId = linkGene.innovationId;
      weight = linkGene.weight;
      enabled = linkGene.enabled;
    }
    
    return *this;
  }

  /**
   * Set fromNeuronId.
   */
  void FromNeuronId(int fromNeuronId) { this->fromNeuronId = fromNeuronId; } 

  /**
   * Get fromNeuronId.
   */
  int FromNeuronId() const { return fromNeuronId; }

  /**
   * Set toNeuronId.
   */
  void ToNeuronId(int toNeuronId) { this->toNeuronId = toNeuronId; }

  /**
   * Get toNeuronId.
   */
  int ToNeuronId() const { return toNeuronId; }

  /**
   * Set innovationId.
   */
  void InnovationId(int innovationId) { this->innovationId = innovationId; }

  /**
   * Get innovationId.
   */
  int InnovationId() const { return innovationId; }

  /**
   * Set weight.
   */
  void Weight(double weight) { this->weight = weight; }

  /**
   * Get weight.
   */
  double Weight() const { return weight; }

  /**
   * Set enabled.
   */
  void Enabled(bool enabled) { this->enabled = enabled; }

  /**
   * Get enabled.
   */
  bool Enabled() const { return enabled; }

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(fromNeuronId, "fromNeuronId");
    ar & data::CreateNVP(toNeuronId, "toNeuronId");
    ar & data::CreateNVP(weight, "weight");
    ar & data::CreateNVP(innovationId, "innovationId");
    ar & data::CreateNVP(enabled, "enabled");
  }

/**
 * Non-intrusive serialization for Neighbor Search class. We need this
 * definition because we are going to use the serialize function for boost
 * variant, which will look for a serialize function for its member types.
 */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    Serialize(ar, version);
  }

 private:
  //! The id of neuron that this link starts from.
  int fromNeuronId;

  //! The id of neuron that this link ends to.
  int toNeuronId;

  //! Link weight.
  double weight;

  //! Link innovation ID.
  int innovationId;

  //! Enabled or not.
  bool enabled;
};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_LINK_GENE_HPP
