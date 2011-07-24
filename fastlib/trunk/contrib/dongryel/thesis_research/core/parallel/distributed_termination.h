/** @file distributed_termination.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_TERMINATION_H
#define CORE_PARALLEL_DISTRIBUTED_TERMINATION_H

#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <deque>
#include "core/parallel/message_tag.h"

namespace core {
namespace parallel {
class DistributedTermination {

  public:

    class TerminationMessage {
      private:
        int stage_;

        int originating_rank_;

      public:

        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) {

          // Serialize the stage and the rank of the origination
          // process.
          ar & stage_;
          ar & originating_rank_;
        }

        int stage() const {
          return stage_;
        }

        int originating_rank() const {
          return originating_rank_;
        }

        void move_to_next_stage() {
          stage_++;
        }

        void Init(int stage_in, int originating_rank_in) {
          stage_ = stage_in;
          originating_rank_ = originating_rank_in;
        }

        void operator=(const TerminationMessage &message_in) {
          this->Init(message_in.stage(), message_in.originating_rank());
        }

        TerminationMessage(const TerminationMessage &message_in) {
          this->operator=(message_in);
        }

        TerminationMessage(int stage_in, int originating_rank_in) {
          this->Init(stage_in, originating_rank_in);
        }

        TerminationMessage() {
          stage_ = 0;
          originating_rank_ = 0;
        }
    };

  private:

    /** @brief The termination status of all MPI processes
     *         maintained by the current MPI process.
     */
    std::deque<bool> terminated_;

    /** @brief The number of MPI process that are done.
     */
    int termination_count_;

    /** @brief Messages originating from other processes that are
     *         in transit.
     */
    std::vector <
    std::pair <
    TerminationMessage, boost::mpi::request > > messages_in_transit_;

    /** @brief The list of free slots used for forwarding the
     *         messages from other processes.
     */
    std::vector<int> free_slots_for_sending_;

    /** @brief The list of slots that are being used for messages.
     */
    std::vector<int> sending_in_progress_;

    /** @brief The diameter of the hypercube network topology.
     */
    int diameter_;

    /** @brief The termination message originating from this
     *         process.
     */
    std::pair< TerminationMessage, boost::mpi::request > self_message_;

    /** @brief Whether the self message can be sent out.
     */
    bool self_message_is_free_;

    /** @brief The next MPI process to send the self message to.
     */
    int next_self_message_dest_;

  public:

    void set_termination_flag(int rank_in) {
      terminated_[ rank_in ] = true;
    }

    bool can_terminate(boost::mpi::communicator &comm) const {
      return termination_count_ == comm.size();
    }

    void Init(boost::mpi::communicator &comm) {

      // Set the termination information.
      terminated_.resize(comm.size());
      for(int i = 0; i < comm.size(); i++) {
        terminated_[i] = false;
      }
      termination_count_ = 0;

      diameter_ = static_cast<int>(ceil(log2(comm.size())));
      messages_in_transit_.resize(diameter_);
      free_slots_for_sending_.resize(diameter_);
      for(int i = 0; i < diameter_; i++) {
        free_slots_for_sending_[i] = i;
      }

      // Initialize the termination message originating from this
      // process.
      self_message_.first.Init(0, comm.rank());
      self_message_is_free_ = true;
      next_self_message_dest_ = (comm.rank() ^ 1);
    }

    void AsynchForwardTerminationMessages(boost::mpi::communicator &comm) {

      // If the current process is done, then send out the
      // message.
      if(terminated_[ comm.rank()]) {
        if(self_message_is_free_) {
          if(next_self_message_dest_ < comm.size()) {
            self_message_.second =
              comm.isend(
                next_self_message_dest_,
                core::parallel::MessageTag::MPI_PROCESS_DONE,
                self_message_.first);
          }
        }
        else if(self_message_.second.test()) {

          // Try to receive the acknowledgement to free the slot, and
          // move onto the next stage.
          self_message_is_free_ = true;
          self_message_.first.move_to_next_stage();
          next_self_message_dest_ = comm.rank() ^ self_message_.first.stage();
        }
      }

      // Probe from any of the neighbors and forward termination
      // messages appropriately.
      for(int i = 0;
          free_slots_for_sending_.size() > 0 && i < diameter_; i++) {

        int probe_index = comm.rank() ^(1 << i);
        if(probe_index < comm.size()) {
          if(boost::optional< boost::mpi::status> l_status =
                comm.iprobe(
                  probe_index,
                  core::parallel::MessageTag::MPI_PROCESS_DONE)) {

            // Receive the probed message and tally the count.
            TerminationMessage received_message;
            comm.recv(
              l_status->source(),
              core::parallel::MessageTag::MPI_PROCESS_DONE,
              received_message);
            terminated_[ received_message.originating_rank()] = true;
            termination_count_++;

            // Forward the message for the next destination.
            int next_destination = 1 << (received_message.stage() + 1);
            if(next_destination < comm.size()) {

              // Get a free slot and prepare the message.
              int free_slot = free_slots_for_sending_.back();
              free_slots_for_sending_.pop_back();
              messages_in_transit_[free_slot].first = received_message;
              messages_in_transit_[free_slot].first.move_to_next_stage();

              // Issue the asnychronous send.
              messages_in_transit_[free_slot].second =
                comm.isend(
                  next_destination,
                  core::parallel::MessageTag::MPI_PROCESS_DONE,
                  messages_in_transit_[free_slot].first);
              sending_in_progress_.push_back(free_slot);
            }
          }
        }
      }

      // Now probe for acknowledgements from the neighbors that
      // they have received the message.
      for(int i = 0; i < static_cast<int>(sending_in_progress_.size()); i++) {
        int test_slot = sending_in_progress_[i];
        if(messages_in_transit_[test_slot].second.test()) {

          // Free the slot if sending is done.
          sending_in_progress_[i] = sending_in_progress_.back();
          sending_in_progress_.pop_back();
          free_slots_for_sending_.push_back(test_slot);
          i--;
        }
      }
    }
};
}
}

#endif
