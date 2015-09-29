/**
 * @file streaming_decision_tree_main.cpp
 * @author Ryan Curtin
 *
 * A command-line executable that can build a streaming decision tree.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/hoeffding_trees/streaming_decision_tree.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_split.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;

PARAM_STRING_REQ("training_file", "Training dataset file.", "t");
PARAM_STRING("labels_file", "Labels for training dataset.", "l", "");

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  const string trainingFile = CLI::GetParam<string>("training_file");
  const string labelsFile = CLI::GetParam<string>("labels_file");

  arma::mat trainingSet;
  DatasetInfo datasetInfo;
  data::Load(trainingFile, trainingSet, datasetInfo, true);

  arma::Row<size_t> labels;
  data::Load(labelsFile, labels, true);

  // Now create the decision tree.
  StreamingDecisionTree<HoeffdingSplit<>> tree(trainingSet, datasetInfo, labels,
      max(labels) + 1);

  // Great.  Good job team.
}
