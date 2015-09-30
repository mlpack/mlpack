/**
 * @file streaming_decision_tree_main.cpp
 * @author Ryan Curtin
 *
 * A command-line executable that can build a streaming decision tree.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/hoeffding_trees/streaming_decision_tree.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_split.hpp>
#include <stack>

using namespace std;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;

PARAM_STRING_REQ("training_file", "Training dataset file.", "t");
PARAM_STRING("labels_file", "Labels for training dataset.", "l", "");

PARAM_DOUBLE("confidence", "Confidence before splitting (between 0 and 1).",
    "c", 0.95);


int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  const string trainingFile = CLI::GetParam<string>("training_file");
  const string labelsFile = CLI::GetParam<string>("labels_file");

  arma::mat trainingSet;
  DatasetInfo datasetInfo;
  data::Load(trainingFile, trainingSet, datasetInfo, true);
  for (size_t i = 0; i < trainingSet.n_rows; ++i)
    Log::Info << datasetInfo.NumMappings(i) << " mappings in dimension " << i <<
".\n";

  arma::Col<size_t> labelsIn;
  data::Load(labelsFile, labelsIn, true, false);
  arma::Row<size_t> labels = labelsIn.t();

  // Now create the decision tree.
  StreamingDecisionTree<HoeffdingSplit<>> tree(trainingSet, datasetInfo, labels,
      max(labels) + 1);

  // Great.  Good job team.
  std::stack<StreamingDecisionTree<HoeffdingSplit<>>*> stack;
  stack.push(&tree);
  while (!stack.empty())
  {
    StreamingDecisionTree<HoeffdingSplit<>>* node = stack.top();
    stack.pop();

    Log::Info << "Node:\n";
    Log::Info << "  split dimension " << node->Split().SplitDimension()
        << ".\n";
    Log::Info << "  majority class " << node->Split().Classify(arma::vec())
        << ".\n";
    Log::Info << "  children " << node->NumChildren() << ".\n";

    for (size_t i = 0; i < node->NumChildren(); ++i)
      stack.push(&node->Child(i));
  }

  // Check the accuracy on the training set.
  arma::Row<size_t> predictedLabels;
  tree.Classify(trainingSet, predictedLabels);

  size_t correct = 0;
  for (size_t i = 0; i < predictedLabels.n_elem; ++i)
    if (labels[i] == predictedLabels[i])
      ++correct;

  Log::Info << correct << " correct out of " << predictedLabels.n_elem << ".\n";
}
