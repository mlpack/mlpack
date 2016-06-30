/**
 * @file tasks.hpp
 * @author Bang Liu
 *
 * Definition of Population class.
 */
#ifndef MLPACK_METHODS_NE_TASKS_HPP
#define MLPACK_METHODS_NE_TASKS_HPP

#include <cstddef>

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "genome.hpp"

namespace mlpack {
namespace ne {

/**
 * This class defines task xor.
 */
template<typename FitnissFunction = ann::MeanSquaredErrorFunction>
class TaskXor {
 public:
  // Evaluate genome's fitness for this task.
  double EvalFitness(Genome& genome) {
  	// Check input size
  	if (genome.NumInput() != 3 || genome.NumOutput() != 1) {
      printf("number of input is %d, output is %d \n", genome.NumInput(), genome.NumOutput()); //!!! DEBUG
  	  return -1;  // -1 means network structure input/output dimension not correct.
  	}

    // Input, output pairs for evaluate fitness.
  	std::vector<std::vector<double>> inputs;  // TODO: use arma::mat for input.
  	std::vector<double> input1 = {0, 0, 1};
  	std::vector<double> input2 = {0, 1, 1};
  	std::vector<double> input3 = {1, 0, 1};
  	std::vector<double> input4 = {1, 1, 1};
  	inputs.push_back(input1);
  	inputs.push_back(input2);
  	inputs.push_back(input3);
  	inputs.push_back(input4);

  	std::vector<double> outputs;
  	outputs.push_back(1);
  	outputs.push_back(0);
  	outputs.push_back(1);
  	outputs.push_back(0);

  	double fitness = 0;
  	for (int i=0; i<4; ++i) {
      //printf("start activate\n");
  		genome.Activate(inputs[i]);
      //printf("end activate\n");
  		double output = genome.Output()[0];
  	    //fitness += FitnissFunction::Error(output, outputs[i]); incorrect
  	    fitness += pow((output - outputs[i]), 2);  // TODO: revise.
  	}

    return fitness;  // fitness smaller is better. 0 is best.
  }

};

// TODO: other task classes that implements a EvalFitness function.

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_TASKS_HPP
