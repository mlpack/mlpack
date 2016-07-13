/**
 * @file tasks.hpp
 * @author Bang Liu
 *
 * Definition of Population class.
 */
#ifndef MLPACK_METHODS_NE_TASKS_HPP
#define MLPACK_METHODS_NE_TASKS_HPP

#include <cstddef>
#include <cmath>

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
  	assert(genome.NumInput() == 3); 
    assert(genome.NumOutput() == 1);

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
  		genome.Activate(inputs[i]);
  		std::vector<double> output;
      genome.Output(output);
  	  fitness += pow((output[0] - outputs[i]), 2);
  	}

    return fitness;  // fitness smaller is better. 0 is best.
  }

};

/**
 * This class defines task cart pole balancing.
 */
class TaskCartPole {
 public:
  // Mass of cart.
  double mc;

  // Mass of pole.
  double mp;

  // Gravity.
  double g;

  // Half length of the pole.
  double l;

  // Magnitude of control force.
  double F;

  // Integration time stamp.
  double tau;

  // Track limits.
  double track_limit;

  // Angle limits.
  double theta_limit;

  // Number of trials to test genomes.
  ssize_t num_trial;

  // Number of steps in each trial.
  ssize_t num_step;

  // Default constructor.
  TaskCartPole() {
    double track_limit = 2.4;
    double theta_limit = 12 * M_PI / 180.0;
    double g = 9.81;
    double mp = 0.1;
    double mc = 1.0;
    double l = 0.5;
    double F = 10.0;
    double tau = 0.02;
    ssize_t num_trial = 20;
    ssize_t num_step = 200;
  }

  // Parametric constructor.
  TaskCartPole(double t_mc, double t_mp, double t_g, double t_l, double t_F, 
               double t_tau, double t_track_limit, double t_theta_limit,
               ssize_t t_num_trial, ssize_t t_num_step):
               mc(t_mc), mp(t_mp), g(t_g), l(t_l), F(t_F),
               tau(t_tau), track_limit(t_track_limit), theta_limit(t_theta_limit),
               num_trial(t_num_trial), num_step(t_num_step) {}

  // Update status.
  void Action(double action, double& x, double& x_dot, double& theta, double& theta_dot) {
    double force = action * F;  // action is -1 or 1. TODO: or continuous???
    double cos_theta = cos(theta * M_PI / 180.0);
    double sin_theta = sin(theta * M_PI / 180.0);
    double total_mass = mp + mc;
    double temp = (force + mp * l * theta_dot * theta_dot * sin_theta) / total_mass;   
    double theta_acc = (g * sin_theta - cos_theta * temp) / (l * (4.0 / 3.0 - mp * cos_theta * cos_theta / total_mass));    
    double x_acc  = temp - mp * l * theta_acc * cos_theta / total_mass;

    // Update the four state variables using Euler's method.
    x += tau * x_dot;
    x_dot += tau * x_acc;
    theta += tau * theta_dot;
    theta_dot += tau * theta_acc;
  }

  // Evaluate genome's fitness for this task.
  double EvalFitness(Genome& genome) {
    assert(genome.NumInput() == 5); 
    assert(genome.NumOutput() == 1);

    double fitness = 0;  // For this task, bigger is better.
    for (ssize_t trial=0; trial<num_trial; ++trial) {
      // Initialize inputs: x, x_dot, theta, theta_dot. As used by Stanley
      double x = mlpack::math::Random(-2.4, 2.4);
      double x_dot = mlpack::math::Random(-1.0, 1.0);
      double theta = mlpack::math::Random(-0.2, 0.2);
      double theta_dot = mlpack::math::Random(-1.5, 1.5);

      for (ssize_t step=0; step<num_step; ++step) {
        // Convert input into range [0, 1]
        std::vector<double> inputs = {(x + 2.4) / 4.8,
                                      (x_dot + 0.75) / 1.5,  // ??
                                      (theta + theta_limit) / 0.41,
                                      (theta_dot + 1.0) / 2.0,
                                      1};  // TODO: use arma::mat for input.
        genome.Activate(inputs);
        std::vector<double> output;
        genome.Output(output);

        double action = output[0];
        if (output[0] < 0.5) {
          action = -1;
        } else {
          action = 1;
        }
        
        Action(action, x, x_dot, theta, theta_dot);
        fitness += 1;
        if(abs(x)>=track_limit || abs(theta)>=theta_limit) break;
      }
    }

    return 1/fitness;
  }

};

// TODO: other task classes that implements a EvalFitness function.

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_TASKS_HPP
