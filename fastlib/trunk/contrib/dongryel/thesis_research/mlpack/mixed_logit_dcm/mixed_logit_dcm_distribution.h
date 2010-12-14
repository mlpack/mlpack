/** @file mixed_logit_dcm_distribution.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_DCM_DISTRIBUTION_H
#define MLPACK_MIXED_LOGIT_DCM_DCM_DISTRIBUTION_H

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The base abstract class for the distribution that
 *         generates each $\beta$ parameter in mixed logit
 *         models. This distribution is parametrized by $\theta$.
 */
class MixedLogitDCMDistribution {
  public:
    virtual int num_parameters() const = 0;
};
};
};

#endif
