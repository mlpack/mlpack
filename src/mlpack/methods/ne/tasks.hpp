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
#include <queue>
#include <boost/math/special_functions/sign.hpp>

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "genome.hpp"
#include "utils.hpp"

namespace mlpack {
namespace ne {

/**
 * This class defines task xor.
 */
template<typename FitnissFunction = ann::MeanSquaredErrorFunction>
class TaskXor {
 public:
  // Task success or not.
  bool success;

  // Default constructor.
  TaskXor() {
    success = false;
  }

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
    
    if (fitness < 10e-8) {
      success = true;
    }
    return fitness;  // fitness smaller is better. 0 is best.
  }

  // Whether task success or not.
  bool Success() {
    return success;
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
  int num_trial;

  // Number of steps in each trial.
  int num_step;

  // Task success or not.
  bool success;

  // Default constructor.
  TaskCartPole() {
    track_limit = 2.4;
    theta_limit = 12 * M_PI / 180.0;
    g = 9.81;
    mp = 0.1;
    mc = 1.0;
    l = 0.5;
    F = 10.0;
    tau = 0.02;
    num_trial = 20;
    num_step = 200;
    success = false;
  }

  // Parametric constructor.
  TaskCartPole(double t_mc, double t_mp, double t_g, double t_l, double t_F, 
               double t_tau, double t_track_limit, double t_theta_limit,
               int t_num_trial, int t_num_step, bool t_success):
               mc(t_mc), mp(t_mp), g(t_g), l(t_l), F(t_F),
               tau(t_tau), track_limit(t_track_limit), theta_limit(t_theta_limit),
               num_trial(t_num_trial), num_step(t_num_step), success(t_success) {}

  // Update status.
  void Action(double action, double& x, double& x_dot, double& theta, double& theta_dot) {
    double force = action * F;  // action is -1 or 1. TODO: or continuous???
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
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

    //mlpack::math::RandomSeed(1);  // If no seed set, each time the fitness will change.
    double fitness = 0;
    for (int trial=0; trial<num_trial; ++trial) {
      // Initialize inputs: x, x_dot, theta, theta_dot. As used by Stanley.
      double x = mlpack::math::Random(-2.4, 2.4);
      double x_dot = mlpack::math::Random(-1.0, 1.0);
      double theta = mlpack::math::Random(-0.2, 0.2);
      double theta_dot = mlpack::math::Random(-1.5, 1.5);

      for (int step=0; step<num_step; ++step) {
        // Scale input.
        std::vector<double> inputs = {(x + 2.4) / 4.8,
                                      (x_dot + 0.75) / 1.5,
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
        if (abs(x)>=track_limit || abs(theta)>=theta_limit) break;
      }
    }

    fitness = 1 - fitness / (num_trial * num_step);
    if (fitness == 0) {
      success = true;
    }
    return fitness;
  }

  // Whether task success or not.
  bool Success() {
    return success;
  }

};

/**
 * This class defines task double pole balancing.
 */
class TaskDoublePole {
 public:
  // Magnitude of control force.
  double F;

  // Gravity.
  double g;

  // Half length of pole 1.
  double l_1;

  // Half length of pole 2.
  double l_2;

  // Mass of pole 1.
  double mp_1;

  // Mass of pole 2.
  double mp_2;

  // Mass of cart.
  double mc;

  // The coefficient of friction for the hinge. (We set the same for both hinge)
  double mup;
  
  // The coefficient of friction for the cart on the track.
  double muc;

  // Track limits.
  double track_limit;

  // Angle limits.
  double theta_limit;

  // Integration time stamp.
  double tau;

  // Number of steps in each trial.
  int num_step;

  // Random init state or not.
  bool random_init_state;

  // Markov or not (Markov means know velocity).
  bool markov;

  // Task success or not.
  bool success;

  // Default constructor.
  TaskDoublePole() {
    track_limit = 2.4;
    theta_limit = 36 * M_PI / 180.0;  
    F = 10.0;
    g = -9.8;
    l_1 = 0.5;
    l_2 = 0.05;
    mc = 1.0;
    mp_1 = 0.1;
    mp_2 = 0.01;
    mup = 0.000002;
    muc = 0;
    num_step = 100000;
    tau = 0.01;
    random_init_state = false;
    markov = true;
    success = false;
  }

  // Parametric constructor.
  TaskDoublePole(bool is_markov) {
    track_limit = 2.4;
    theta_limit = 36 * M_PI / 180.0;  
    F = 10.0;
    g = -9.8;
    l_1 = 0.5;
    l_2 = 0.05;
    mc = 1.0;
    mp_1 = 0.1;
    mp_2 = 0.01;
    mup = 0.000002;
    muc = 0;
    num_step = 100000;
    tau = 0.01;
    random_init_state = false;
    markov = is_markov;
    success = false;
  }

  /**
   * Calculate derivatives of 6 states.
   * state = [x, x', theta1, theta1', theta2, theta2'].
   * state_dot = [x', x'', theta1', theta1'', theta', theta2'']
   * The result will be used in RK4.
   */
  void step(double action, const std::vector<double>& state, std::vector<double>& state_dot) {
    // Pre-calculate some value to accelerate program.
    double force = action * F;  // action is -1 or 1.
    double costheta_1 = cos(state[2]);
    double sintheta_1 = sin(state[2]);
    double gsintheta_1 = g * sintheta_1;
    double costheta_2 = cos(state[4]);
    double sintheta_2  = sin(state[4]);
    double gsintheta_2 = g * sintheta_2;
    double ml_1   = mp_1 * l_1;
    double ml_2   = mp_2 * l_2;
    double temp_1 = mup * state[3] / ml_1;
    double temp_2 = mup * state[5] / ml_2;
    
    double fi_1 = (ml_1 * state[3] * state[3] * sintheta_1) +
                  (0.75 * mp_1 * costheta_1 * (temp_1 + gsintheta_1));
    double fi_2 = (ml_2 * state[5] * state[5] * sintheta_2) +
                  (0.75 * mp_2 * costheta_2 * (temp_2 + gsintheta_2));

    double mi_1 = mp_1 * (1 - (0.75 * costheta_1 * costheta_1));
    double mi_2 = mp_2 * (1 - (0.75 * costheta_2 * costheta_2));
    
    // Calculate derivatives of states.
    state_dot[0] = state[1];
    state_dot[2] = state[3];
    state_dot[4] = state[5];
    state_dot[1] = (force - muc * sgn<double>(state[1]) + fi_1 + fi_2) / (mi_1 + mi_2 + mc);
    state_dot[3] = -0.75 * (state_dot[1] * costheta_1 + gsintheta_1 + temp_1) / l_1;
    state_dot[5] = -0.75 * (state_dot[1] * costheta_2 + gsintheta_2 + temp_2) / l_2;
  }

  /**
   * Use Runge-Kutta 4th order methods to update states.
   */
  void RK4(double action, std::vector<double>& state, const std::vector<double>& state_dot) {
    double h2 = 0.5 * tau;
    double h6 = tau / 6.0;
    std::vector<double> dym(6);
    std::vector<double> dyt(6);
    std::vector<double> yt(6);

    for (int i = 0; i < 6; ++i) { 
      yt[i] = state[i] + h2 * state_dot[i];
    }

    step(action, yt, dyt);

    for (int i = 0; i < 6; ++i) {
      yt[i] = state[i]+ h2 * dyt[i];
    }

    step(action, yt, dym);

    for (int i = 0; i < 6; ++i) {
      yt[i] = state[i] + tau * dym[i];
      dym[i] += dyt[i];
    }

    step(action, yt, dyt);

    for (int i = 0; i < 6; ++i) {
      state[i] += h6 * (state_dot[i] + dyt[i] + 2.0 * dym[i]);
    }
  }

  /**
   * Run action for a step to update state.
   */
  void Action(double action, std::vector<double>& state, std::vector<double>& state_dot) {
    for (int i = 0; i < 2; ++i) {
      step(action, state, state_dot);
      RK4(action, state, state_dot);
    }
  }

  // Initialize double pole balancing system states.
  void InitState(bool rand, std::vector<double>& state) {
    state.clear();

    if (rand) {
      state.push_back(mlpack::math::RandInt(0, 5000) / 1000.0 - 2.4);  // cart's initial position x
      state.push_back(mlpack::math::RandInt(0, 2000) / 1000.0 - 1.0);  // cart's initial speed x_dot
      state.push_back(mlpack::math::RandInt(0, 400) / 1000.0 - 0.2);  // pole_1 initial angle theta1
      state.push_back(mlpack::math::RandInt(0, 400) / 1000.0 - 0.2);  // pole_1 initial angular velocity theta1_dot
      state.push_back(mlpack::math::RandInt(0, 3000) / 1000.0 - 0.4);  // pole_2 initial angle theta2
      state.push_back(mlpack::math::RandInt(0, 3000) / 1000.0 - 0.4);  // pole_2 initial angular velocity theta2_dot
    } else {
      state.push_back(0);
      state.push_back(0);
      state.push_back(0.07);  // set pole_1 to one degree (in radians)
      state.push_back(0);
      state.push_back(0);
      state.push_back(0);
    }
  }

  // Whether cart outside bounds.
  bool OutsideBounds(std::vector<double>& state) {
    return (abs(state[0]) >= track_limit ||
            abs(state[2]) >= theta_limit ||
            abs(state[4]) >= theta_limit);
  }

  // Markovian: velocity information is provided to the network input.
  double EvalMarkov(Genome& genome, int numStep) {
    assert(genome.NumInput() == 7); // 6 state input + 1 bias.
    assert(genome.NumOutput() == 1);
    
    std::vector<double> state(6);
    InitState(random_init_state, state);
    std::vector<double> state_dot(6);

    int step = 0;
    while (step < numStep) {
      // Input normalized states to genome and get output action.
      std::vector<double> inputs = { state[0] / 4.80, state[1] / 2.00, state[2] / 0.52,
                                     state[3] / 2.00, state[4] / 0.52, state[5] / 2.00,
                                     1 };
      genome.Activate(inputs);
      std::vector<double> output;
      genome.Output(output);    
      double action = output[0];
      if (output[0] < 0.5) {
        action = -1;
      } else {
        action = 1;
      }

      // Update states with action: advances one time step.
      Action(action, state, state_dot);
      if (OutsideBounds(state)) break;
      step += 1;
    }

    return step;
  }

  // Non-Markovian: no velocity is provided.
  double EvalNonMarkov(Genome& genome, int numStep, double& GuruFitness) {
    assert(genome.NumInput() == 4); // 3 state input + 1 bias. No velocity inputs.
    assert(genome.NumOutput() == 1);
    
    std::vector<double> state(6);
    InitState(random_init_state, state);
    std::vector<double> state_dot(6);

    std::queue<double> lastValues;

    int step = 0;
    while (step < numStep) {
      // Input normalized states to genome and get output action.
      std::vector<double> inputs = { state[0] / 4.80, state[2] / 0.52, state[4] / 0.52, 1 };
      genome.Activate(inputs);
      std::vector<double> output;
      genome.Output(output);    
      double action = output[0];
      if (output[0] < 0.5) {
        action = -1;
      } else {
        action = 1;
      }

      // Update states with action: advances one time step.
      Action(action, state, state_dot);
      if (OutsideBounds(state)) break;

      // To calculate Gruau's fitness.
      double value = abs(state[0]) + abs(state[1]) + abs(state[2]) + abs(state[3]);
      lastValues.push(value);
      if (lastValues.size() == 101) lastValues.pop();  // keep last 100 values.

      step += 1;
    }

    // Calculate Gruau's fitness.
    if (step >= 100) {
      double jiggle = 0;
      while (!lastValues.empty()) {
        jiggle += lastValues.front();
        lastValues.pop();
      }

      GuruFitness = 0.1 * step / 1000.0 + 0.9 * 0.75 / jiggle;  // Currently, it is useless.
    } else {
      GuruFitness = 0.1 * step / 1000.0;
    }
    
    return step;
  }

  // Generalization test. Test 625 different initial states.
  // Return how many initials states can be balanced.
  int GeneralizationTest(Genome& genome) {
    std::vector<double> stateEvals = {0.05, 0.25, 0.5, 0.75, 0.95};

    int balanced = 0;
    int testNumber = 0;
    for (int i1 = 0; i1 < 5; ++i1) {
      for (int i2 = 0; i2 < 5; ++i2) {
        for (int i3 = 0; i3 < 5; ++i3) {
          for (int i4 = 0; i4 < 5; ++i4) {
            double x = stateEvals[i1] * 4.32 - 2.16;
            double x_dot = stateEvals[i2] * 2.70 - 1.35;
            double theta1 = stateEvals[i3] * 0.12566304 - 0.06283152;
            double theta1_dot = stateEvals[i4] * 0.30019504 - 0.15009752;
            std::vector<double> state = {x, x_dot, theta1, theta1_dot, 0, 0};

            testNumber += 1;

            double GuruFitness = 0;
            double score = EvalNonMarkov(genome, 1000, GuruFitness);
            if (score > 999) balanced += 1;
          }
        }
      }
    }

    return balanced;
  }

  // Evaluate fitness of a genome.
  double EvalFitness(Genome& genome) {
    double fitness = DBL_MAX;

    if (markov) {
      fitness = EvalMarkov(genome, 100000);
    } else {
      double GuruFitness = 0;
      double score = EvalNonMarkov(genome, 100000, GuruFitness);
      fitness = score;
      
      // If passed 100000 step testing, continue with generalization testing.
      if (score > 99999) {
        int balanced = GeneralizationTest(genome);

        if (balanced > 200) {
          fitness = 100000;
        }
      }
    }

    if (fitness == 100000) {
      success = true;
    }

    fitness = -fitness;
    return fitness;
  }

  // Whether task success or not.
  bool Success() {
    return success;
  }

};

/**
 * This class defines task mountain car.
 */
class TaskMountainCar {
 public:
  // Low bound of position.
  double x_l;

  // High bound of position.
  double x_h;

  // Low bound of velocity.
  double x_dot_l;

  // High bound of velocity.
  double x_dot_h;

  // Acceleration due to gravity.
  double gravity;

  // Above this value means goal reached.
  double goal;

  // Number of trials to test genomes.
  int num_trial;

  // Number of steps in each trial.
  int num_step;

  // Task success or not.
  bool success;

  // Default constructor.
  TaskMountainCar() {
    x_l = -1.2;
    x_h = 0.5;
    x_dot_l = -0.07;
    x_dot_h = 0.07;
    gravity = -0.0025;
    goal = 0.5;
    num_trial = 20;
    num_step = 200;
    success = false;
  }

  // Parametric constructor.
  TaskMountainCar(double t_x_l, double t_x_h, double t_x_dot_l,
                  double t_x_dot_h, double t_gravity, double t_goal,
                  double t_num_trial, double t_num_step, bool t_success):
                  x_l(t_x_l), x_h(t_x_h), x_dot_l(t_x_dot_l),
                  x_dot_h(t_x_dot_h), gravity(t_gravity), goal(t_goal),
                  num_trial(t_num_trial), num_step(t_num_step), success(t_success) {}

  // Update status.
  void Action(double action, double& x, double& x_dot) {
    x_dot = x_dot + 0.001 * action + (gravity * cos(3 * x));
    if (x_dot < x_dot_l) x_dot = x_dot_l;
    if (x_dot > x_dot_h) x_dot = x_dot_h;

    x = x + x_dot;
    if (x < x_l) {
      x = x_l;
      x_dot = 0;
    }
  }

  // Evaluate genome's fitness for this task.
  double EvalFitness(Genome& genome) {
    assert(genome.NumInput() == 3); 
    assert(genome.NumOutput() == 3);

    double fitness = 0;
    int numSuccess = 0;
    for (int trial=0; trial<num_trial; ++trial) {
      // Initialize inputs: x, x_dot.
      double x = mlpack::math::Random(x_l, x_h);
      double x_dot = mlpack::math::Random(x_dot_l, x_dot_h);

      for (int step=0; step<num_step; ++step) {
        // Get action.
        std::vector<double> inputs = {x, x_dot, 1};
        genome.Activate(inputs);
        std::vector<double> output;
        genome.Output(output);

        double action = 0;
        auto biggest_position = std::max_element(std::begin(output), std::end(output));
        int index = std::distance(std::begin(output), biggest_position);
        if (index == 0) action = 0;
        if (index == 1) action = 1;
        if (index == 2) action = -1; // NOTICE: we haven't consider equal cases. two equal or three equal.
        
        // Update position x and velocity x_dot.
        Action(action, x, x_dot);

        // Update fitness.
        fitness += 1;
        if (x >= goal) {
          numSuccess += 1;
          break;
        }
      }
    }
    
    if (numSuccess == num_trial) {
      success = true;
    }
    
    fitness = 1 / fitness;
    return fitness;
  }

  // Whether task success or not.
  bool Success() {
    return success;
  }

};

// TODO: other task classes that implements a EvalFitness function.

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_TASKS_HPP
