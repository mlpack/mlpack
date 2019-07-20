/**
 * @file neat_main.cpp
 * @author Rahul Ganesh Prabhu
 *
 * Executable for running NEAT.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "neat.hpp"

using namespace mlpack;
using namespace mlpack::neat;
using namespace std;

// Define parameters for the executable.
PROGRAM_INFO("NeuroEvolution of Augmenting Topologies",
		// Short description.
		"An implementation of NeuroEvolution of Augmenting Topologies which, "
		"given a task and basic parameters, will return a simple neural network "
		"with the ability to solve the task.",
		// Long description.
		"This program uses NeuroEvolution of Augmenting Topologies to find a simple"
		" neural network that can perform a given task."
		"\n\n"
		"");

// Required options.
PARAM_INT_REQ("input_node_count", "Number of input nodes", "i");
PARAM_INT_REQ("output_node_count", "Number of output nodes", "o");
PARAM_INT_REQ("pop_size", "Population size", "p");
PARAM_INT_REQ("max_gen", "Maximum number of generations", "g");
PARAM_INT_REQ("num_species", "Number of species", "s");

// NEAT configuration options.
PARAM_DOUBLE_IN("initial_bias", "The bias with which genomes are initialized",
		"B", 1.0);
PARAM_DOUBLE_IN("initial_weight", "The initial connection weights", "W", 0.0);
PARAM_DOUBLE_IN("weight_mut_prob", "The probability of a weight being mutated",
		"w", 0.8);
PARAM_DOUBLE_IN("weight_mut_size", "The degree of mutation of weights", "d",
		0.5);
PARAM_DOUBLE_IN("bias_mut_prob", "The probability of a bias being mutated",
		"b", 0.7);
PARAM_DOUBLE_IN("bias_mut_size", "The degree of mutation of bias", "D", 0.5);
PARAM_DOUBLE_IN("node_add_prob", "The probability of node addition", "n", 0.2);
PARAM_DOUBLE_IN("conn_add_prob", "The probability of connection addition", "c",
		0.5);
PARAM_DOUBLE_IN("conn_del_prob", "The probability of connection deletion", "C",
		0.5);
PARAM_DOUBLE_IN("disable_prob", "The probability of a disabled gene becoming "
		"enabled during crossover", "P", 0.2);
PARAM_DOUBLE_IN("elitism_prop", "The proportion of a species that is considered"
		" elite", "e", 0.1);
PARAM_DOUBLE_IN("final_fitness", "The desired fitness of the genomes. If it is "
		"zero, no limit on the fitness is considered", "f", 0);
PARAM_INT_IN("complexity_threshold", "The maximum complexity allowed", "t", 0);
PARAM_FLAG("is_acyclic", "Denotes whether or not the genomes are meant to be ",
		"acyclic", "a", false);

static void mlpackMain()
{
	
}



