/**
 * @file methods/grad_boosting/grad_boosting.hpp
 * @author Abhimanyu Dayal
 *
 * Gradient Boosting class. 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_HPP
#define MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>

namespace mlpack {
/**
 * The Gradient Boosting class. Gradient Boosting is a boosting algorithm, meaning that it
 * combines an ensemble of weak learners to produce a strong learner.
 * 
 * Gradient Boosting is generally implemented using Decision Trees, or more specifically 
 * Decision Stumps i.e. weak learner Decision Trees with low depth.
 * 
 * @tparam MatType Data matrix type (i.e. arma::mat or arma::sp_mat).
 */

template<typename WeakLearnerType = ID3DecisionStump<>, 
        typename MatType = arma::mat>
class GradBoosting {
    public: 
    
    /**
     * Constructor for creating GradBoosting without training. 
     * Be sure to call Train() before calling Classify()
     */
    GradBoosting();

    /**
     * Constructor for a GradBoosting model. Any extra parameters are used as
     * hyperparameters for the weak learner. These should be the last arguments
     * to the weak learner's constructor or `Train()` function (i.e. anything
     * after `numClasses` or `weights`).
     *
     * @param data Input data.
     * @param labels Corresponding labels.
     * @param numClasses The number of classes.
     * @param num_models Number of weak learners.
     * @param weakLearnerParams... Any hyperparameters for the weak learner.
     */
    template<typename... WeakLearnerArgs>
    GradBoosting(
        const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t num_models = 10,
        WeakLearnerArgs&&... weakLearnerArgs
    );

    /**
     * Constructor takes an already-initialized weak learner; all other
     * weak learners will learn with the same parameters as the given
     * weak learner.
     *
     * @param data Input data.
     * @param labels Corresponding labels.
     * @param numClasses The number of classes.
     * @param num_models Number of weak learners.
     * @param other Weak learner that has already been initialized.
     */
    template<typename WeakLearnerInType>
    GradBoosting (
        const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t num_models = 10,
        const WeakLearnerInType& other,
        const typename std::enable_if<
            std::is_same<WeakLearnerType, WeakLearnerInType>::value
        >::type* = 0
    );

    //! Get the number of classes this model is trained on.
    size_t NumClasses() const { return numClasses; }

    //! Get the number of weak learners .
    size_t NumModels() const { return num_models; }

    //! Get the weights for the given weak learner.
    ElemType Alpha(const size_t i) const { return alpha[i]; }

    //! Modify the weight for the given weak learner (be careful!).
    ElemType& Alpha(const size_t i) { return alpha[i]; }

    //! Get the given weak learner.
    const WeakLearnerType& WeakLearner(const size_t i) const { return wl[i]; }
    
    //! Modify the given weak learner (be careful!).
    WeakLearnerType& WeakLearner(const size_t i) { return wl[i]; }

    /**
     * Train GradBoosting on the given dataset. This method takes an initialized
     * WeakLearnerType; the parameters for this weak learner will be used to train
     * each of the weak learners during GradBoosting training. Note that this will
     * completely overwrite any model that has already been trained with this
     * object.
     *
     * Default values are not used for `num_models`; instead, it is used to specify
     * the number of weak learners (models) to train during gradient boosting.
     *
     * @param data Dataset to train on.
     * @param labels Labels for each point in the dataset.
     * @param numClasses The number of classes.
     * @param learner Learner to use for training.
     * @param num_models Number of weak learners (models) to train.
     */
    template<typename WeakLearnerInType>
    void Train(
        const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const WeakLearnerInType& learner,
        const size_t num_models
    );

    /**
     * Train Gradient Boosting on the given dataset, using the given parameters.
     * The last parameters are the hyperparameters to use for the weak learners;
     * these are all the arguments to `WeakLearnerType::Train()` after `numClasses`
     * and `weights`.
     *
     * Default values are not used for `num_models`; instead, it is used to specify
     * the number of weak learners (models) to train during gradient boosting.
     *
     * @param data Dataset to train on.
     * @param labels Labels for each point in the dataset.
     * @param numClasses The number of classes in the dataset.
     * @param num_models Number of boosting rounds.
     * @param weakLearnerArgs Hyperparameters to use for each weak learner.
     */
    ElemType Train(
        const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t num_models,
        const typename std::enable_if<
            std::is_same<WeakLearnerType, WeakLearnerInType>::value>::type* = 0
    );

    template<typename... WeakLearnerArgs>
    ElemType Train(
        const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t num_models,
        WeakLearnerArgs&&... weakLearnerArgs
    );

    /**
     * Classify the given test point.
     *
     * @param point Test point.
     */
    template<typename VecType>
    size_t Classify(const VecType& point) const;

    /**
     * Classify the given test point and compute class probabilities.
     *
     * @param point Test point.
     * @param prediction Will be filled with the predicted class of `point`.
     * @param probabilities Will be filled with the class probabilities.
     */
    template<typename VecType>
    void Classify(const VecType& point,
                    size_t& prediction,
                    arma::Row<ElemType>& probabilities) const;

    /**
     * Classify the given test points.
     *
     * @param test Testing data.
     * @param predictedLabels Vector in which the predicted labels of the test
     *      set will be stored.
     */
    void Classify(const MatType& test,
                    arma::Row<size_t>& predictedLabels) const;

    /**
     * Classify the given test points.
     *
     * @param test Testing data.
     * @param predictedLabels Vector in which the predicted labels of the test
     *      set will be stored.
     * @param probabilities matrix to store the predicted class probabilities for
     *      each point in the test set.
     */
    void Classify(const MatType& test,
                    arma::Row<size_t>& predictedLabels,
                    arma::Mat<ElemType>& probabilities) const;
    

    /**
     * Serialize the GradBoosting model.
     */
    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */);

    private:
    /**
     * Internal utility training function.  `wl` is not used if
     * `UseExistingWeakLearner` is false.  `weakLearnerArgs` are not used if
     * `UseExistingWeakLearner` is true.
     */
    template<bool UseExistingWeakLearner, typename... WeakLearnerArgs>
    ElemType TrainInternal(const MatType& data,
                            const arma::Row<size_t>& labels,
                            const size_t numClasses,
                            const WeakLearnerType& wl,
                            WeakLearnerArgs&&... weakLearnerArgs);

    //! The number of classes in the model.
    size_t numClasses;
    //! The number of weak learners in the model.
    size_t num_models;

    //! The vector of weak learners.
    std::vector<WeakLearnerType> wl;
    //! The weights corresponding to each weak learner.
    std::vector<ElemType> alpha;
}; 

}

CEREAL_TEMPLATE_CLASS_VERSION((typename WeakLearnerType, typename MatType),
    (mlpack::GradBoosting<WeakLearnerType, MatType>), (1));

// Include implementation.
#include "grad_boosting_impl.hpp"

#endif
