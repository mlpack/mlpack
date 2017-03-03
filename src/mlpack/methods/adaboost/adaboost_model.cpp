/**
 * @file adaboost_model.cpp
 * @author Ryan Curtin
 *
 * A serializable AdaBoost model, used by the main program.
 */
#include "adaboost_model.hpp"

using namespace mlpack;
using namespace std;
using namespace arma;
using namespace mlpack::adaboost;
using namespace mlpack::decision_stump;
using namespace mlpack::perceptron;

//! Create an empty AdaBoost model.
AdaBoostModel::AdaBoostModel() :
    weakLearnerType(0),
    dsBoost(NULL),
    pBoost(NULL),
    dimensionality(0)
{
  // Nothing to do.
}

//! Create the AdaBoost model with the given mappings and type.
AdaBoostModel::AdaBoostModel(
    const Col<size_t>& mappings,
    const size_t weakLearnerType) :
    mappings(mappings),
    weakLearnerType(weakLearnerType),
    dsBoost(NULL),
    pBoost(NULL),
    dimensionality(0)
{
  // Nothing to do.
}

//! Copy constructor.
AdaBoostModel::AdaBoostModel(const AdaBoostModel& other) :
    mappings(other.mappings),
    weakLearnerType(other.weakLearnerType),
    dsBoost(other.dsBoost == NULL ? NULL :
        new AdaBoost<DecisionStump<>>(*other.dsBoost)),
    pBoost(other.pBoost == NULL ? NULL :
        new AdaBoost<Perceptron<>>(*other.pBoost)),
    dimensionality(other.dimensionality)
{
  // Nothing to do.
}

//! Move constructor.
AdaBoostModel::AdaBoostModel(AdaBoostModel&& other) :
    mappings(std::move(other.mappings)),
    weakLearnerType(other.weakLearnerType),
    dsBoost(other.dsBoost),
    pBoost(other.pBoost),
    dimensionality(other.dimensionality)
{
  other.weakLearnerType = 0;
  other.dsBoost = NULL;
  other.pBoost = NULL;
  other.dimensionality = 0;
}

//! Copy assignment operator.
AdaBoostModel& AdaBoostModel::operator=(const AdaBoostModel& other)
{
  mappings = other.mappings;
  weakLearnerType = other.weakLearnerType;

  delete dsBoost;
  dsBoost = (other.dsBoost == NULL) ? NULL :
      new AdaBoost<DecisionStump<>>(*other.dsBoost);

  delete pBoost;
  pBoost = (other.pBoost == NULL) ? NULL :
      new AdaBoost<Perceptron<>>(*other.pBoost);

  dimensionality = other.dimensionality;

  return *this;
}

AdaBoostModel::~AdaBoostModel()
{
  delete dsBoost;
  delete pBoost;
}

//! Train the model.
void AdaBoostModel::Train(const mat& data,
                          const Row<size_t>& labels,
                          const size_t iterations,
                          const double tolerance)
{
  dimensionality = data.n_rows;
  if (weakLearnerType == WeakLearnerTypes::DECISION_STUMP)
  {
    delete dsBoost;

    DecisionStump<> ds(data, labels, max(labels) + 1);
    dsBoost = new AdaBoost<DecisionStump<>>(data, labels, ds, iterations,
        tolerance);
  }
  else if (weakLearnerType == WeakLearnerTypes::PERCEPTRON)
  {
    Perceptron<> p(data, labels, max(labels) + 1);
    pBoost = new AdaBoost<Perceptron<>>(data, labels, p, iterations,
        tolerance);
  }
}

//! Classify test points.
void AdaBoostModel::Classify(const mat& testData, Row<size_t>& predictions)
{
  if (weakLearnerType == WeakLearnerTypes::DECISION_STUMP)
    dsBoost->Classify(testData, predictions);
  else if (weakLearnerType == WeakLearnerTypes::PERCEPTRON)
    pBoost->Classify(testData, predictions);
}
