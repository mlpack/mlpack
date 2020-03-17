#ifndef MLPACK_METHODS_RL_REPLAY_EPISODIC_REPLAY_HPP
#define MLPACK_METHODS_RL_REPLAY_EPISODIC_REPLAY_HPP

#include <mlpack/prereqs.hpp>
#include <random>

namespace mlpack {
namespace rl {

/**
 * Implementation of episodic experience replay.
 *
 * Each episode of interactions between the agent and the
 * environment will be saved to a memory buffer. When necessary,
 * we can simply sample previous episodes or the most recent episode
 * from the buffer to train the agent.
 * @tparam EnvironmentType Desired task.
 */
template <typename EnvironmentType>
class EpisodicReplay
{
 public:
  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  EpisodicReplay():
      capacity(0),
      position(0),
      full(false)
  { /* Nothing to do here. */ }

  /**
  * Construct an instance of episode experience replay class.
  *
  * @param capacity Total memory size in terms of number of episodes.
  * @param dimension The dimension of an encoded state.
  */

  EpisodicReplay(const size_t capacity,
                 const size_t dimension = StateType::dimension) :
      capacity(capacity),
      states.resize(capacity),
      next_states.resize(capacity),
      rewards.resize(capacity),
      actions.resize(capacity),
      isTerminal.resize(capacity),
      position(0),
      full(false)
  { /* Nothing to do here. */ }


  /**
  * Store the given experience.
  *
  * @param state Given state.
  * @param action Given action.
  * @param reward Given reward.
  * @param nextState Given next state.
  * @param isEnd Whether next state is terminal state.
  */
  void Store(const StateType& state,
	     ActionType action,
	     double reward,
	     const StateType& nextState,
	     bool isEnd)
  {
    states[position].push_back(state.Encode());
    actions[position].push_back(action);
    rewards[position].push_back(reward);
    nextStates[position].push_back(nextState.Encode());
    isTerminal[position].push_back(isEnd);
    if(isTerminal){
      position++;
    }
    if (position == capacity)
    {
      full = true;
      position = 0;
    }
  }

  /**
  * Sample some episodes.
  *
  * @param sampledStates Sampled encoded states.
  * @param sampledActions Sampled actions.
  * @param sampledRewards Sampled rewards.
  * @param sampledNextStates Sampled encoded next states.
  * @param isTerminal Indicate whether corresponding next state is terminal
  *        state.
  */

  void Random_Episode(arma::mat& episodeStates,
                      arma::icolvec& episodeActions,
	              arma::colvec& episodeRewards,
	              arma::mat& episodeNextStates,
	              arma::icolvec& isTerminal)
  {
    size_t upperBound = full ? capacity : position;
    srand(time(NULL));
    int episodeNum = rand()%(upperBound);
    vector<arma::colvec> temp = states[episodeNum];
    int i=0;
    for(auto state : temp){
      if(i==0){
        episodeStates = state;
        i++;
      }else{
        episodeStates = arma::join_rows(episodeStates,state);
      }
    }
    episodeActions = arma::conv_to<arma::icolvec>::from(actions[episodeNum]);
    episodeRewards = arma::conv_to<arma::icolvec>::from(rewards[episodeNum]);
    temp = next_states[episodeNum];
    int i=0;
    for(auto state : temp){
      if(i==0){
        episodeNextStates = state;
        i++;
      }else{
        episodeNextStates = arma::join_rows(episodeStates,state);
      }
    }
    isTerminal = arma::conv_to<arma::icolvec>::from(this->isTerminal[episodeNum]);
    }

  /**
  * Get the number of episodes in the memory.
  *
  * @return Actual used memory size
  */

  const size_t& Size()
  {
    return full ? capacity : position;
  }


  /**
  * Get the most recently added episode.
  *
  * @param sampledStates Sampled encoded states.
  * @param sampledActions Sampled actions.
  * @param sampledRewards Sampled rewards.
  * @param sampledNextStates Sampled encoded next states.
  * @param isTerminal Indicate whether corresponding next state is terminal
  *        state.
  */
  void Recent_Episode(arma::mat& episodeStates,
                      arma::icolvec& episodeActions,
                      arma::colvec& episodeRewards,
                      arma::mat& episodeNextStates,
                      arma::icolvec& isTerminal)
  {
    int episodeNum = position;
    vector<arma::colvec> temp = states[episodeNum];
    int i=0;
    for(auto state : temp){
      if(i==0){
        episodeStates = state;
        i++;
      }else{
        episodeStates = arma::join_rows(episodeStates,state);
      }
    }
    episodeActions = arma::conv_to<arma::icolvec>::from(actions[episodeNum]);
    episodeRewards = arma::conv_to<arma::icolvec>::from(rewards[episodeNum]);
    temp = next_states[episodeNum];
    int i=0;
    for(auto state : temp){
      if(i==0){
        episodeNextStates = state;
        i++;
      }else{
        episodeNextStates = arma::join_rows(episodeStates,state);
      }
    }
    isTerminal = arma::conv_to<arma::icolvec>::from(this->isTerminal[episodeNum]);	
  }


 private:
  //! Locally-stored total episode limit.
  size_t capacity;

  //! Indicate the position to store new episode.
  size_t position;

  //! Locally-stored encoded previous states.
  std::vector< std::vector< arma::colvec> > states;

  //! Locally-stored previous actions.
  std::vector< std::vector<int> > actions;

  //! Locally-stored previous rewards.
  std::vector< std::vector<double> > rewards;

  //! Locally-stored encoded previous next states.
  std::vector< std::vector< arma::colvec> > next_states;

  //! Locally-stored termination information of previous experience.
  std::vector< std::vector<int> > isTerminal;

  //! Locally-stored indicator that whether the memory is full or not
  bool full;
};

} // namespace rl
} // namespace mlpack

#endif
