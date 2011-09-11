/** @file kde_statistic.h
 *
 *  The statistics computed from the data. This is stored in each node
 *  of the tree.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_STATISTIC_H
#define MLPACK_KDE_KDE_STATISTIC_H

namespace mlpack {
namespace kde {

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
class KdeStatistic {

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

  public:

    mlpack::series_expansion::CartesianFarField <
    ExpansionType > farfield_expansion_;

    mlpack::kde::KdePostponed<ExpansionType> postponed_;

    mlpack::kde::KdeSummary summary_;

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
      ar & farfield_expansion_;
      ar & postponed_;
      ar & summary_;
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      ar & farfield_expansion_;
      ar & postponed_;
      ar & summary_;

      // Initialize the local expansion to be an empty one.
      postponed_.local_expansion_.Init(
        farfield_expansion_.get_center(),
        farfield_expansion_.get_coeffs().n_elem);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief Copies another KDE statistic.
     */
    void Copy(const KdeStatistic &stat_in) {
      farfield_expansion_.Copy(stat_in.farfield_expansion_);
      postponed_.Copy(stat_in.postponed_);
      summary_.Copy(stat_in.summary_);
    }

    /** @brief The default constructor.
     */
    KdeStatistic() {
      SetZero();
    }

    /** @brief Sets the postponed and the summary statistics to zero.
     */
    void SetZero() {
      postponed_.SetZero();
      summary_.SetZero();
    }

    void Seed(double initial_pruned_in) {
      postponed_.SetZero();
      summary_.Seed(initial_pruned_in);
    }

    /** @brief Initializes by taking statistics on raw data.
     */
    template<typename GlobalType, typename TreeType>
    void Init(const GlobalType &global, TreeType *node) {

      // The node iterator.
      typename GlobalType::TableType::TreeIterator node_it =
        const_cast<GlobalType &>(global).
        reference_table()->get_node_iterator(node);

      // Sets the postponed quantities and summary statistics to zero.
      SetZero();

      // Form the far-field moments.
      arma::vec node_center;
      node->bound().center(&node_center);
      farfield_expansion_.Init(global.kernel_aux(), node_center);
      farfield_expansion_.AccumulateCoeffs(
        global.kernel_aux(), node_it,
        global.kernel_aux().global().get_max_order());

      // Initialize the local expansion.
      postponed_.local_expansion_.Init(global.kernel_aux(), node_center);
    }

    /** @brief Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    template<typename GlobalType, typename TreeType>
    void Init(
      const GlobalType &global,
      TreeType *node,
      const KdeStatistic &left_stat,
      const KdeStatistic &right_stat) {

      // Sets the postponed quantities and summary statistics to zero.
      SetZero();

      // Form the far-field moments.
      arma::vec node_center;
      node->bound().center(&node_center);
      farfield_expansion_.Init(global.kernel_aux(), node_center);
      farfield_expansion_.TranslateFromFarField(
        global.kernel_aux(), left_stat.farfield_expansion_);
      farfield_expansion_.TranslateFromFarField(
        global.kernel_aux(), right_stat.farfield_expansion_);

      // Initialize the local expansion.
      postponed_.local_expansion_.Init(global.kernel_aux(), node_center);
    }
};
}
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
class KdeStatistic;

namespace boost {
namespace serialization {
template<>
template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType >
struct tracking_level <
    mlpack::kde::KdeStatistic<ExpansionType> > {
  typedef mpl::integral_c_tag tag;
  typedef mpl::int_< boost::serialization::track_never > type;
  BOOST_STATIC_CONSTANT(
    int,
    value = tracking_level::type::value
  );
  BOOST_STATIC_ASSERT((
                        mpl::greater <
                        implementation_level< mlpack::kde::KdeStatistic<ExpansionType> >,
                        mpl::int_<primitive_type>
                        >::value
                      ));
};
}
}

#endif
