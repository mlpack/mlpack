// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file mpigrain.h
 *
 * Tool for creating simple MPI parallel programs.
 * You must build with --compiler=mpi.
 *
 * WARNING!  This currently DOES NOT have ANY WAY for you to send
 * results back!
 *
 * TODO: Broken
 */

#ifndef PAR_MPIGRAIN_H
#define PAR_MPIGRAIN_H

#include "xrun/xrun.h"

#include "grain.h"

#include <mpi.h>

/**
 * Grain-based parallelism over MPI.
 *
 * Grains must be bit-copiable.  In the future serialization might be a better
 * idea.
 *
 * Uses single master that maintains priority queue, and handles work out
 * to slaves.
 *
 * TODO: This does NOT have any way for you to send results back to the
 * mater computer.
 */
template<typename TGrain, typename TContext = int>
class MPIGrainRunner {
  FORBID_COPY(MPIGrainRunner);
  friend class MPIDispatcher;

 public:
  typedef TGrain Grain;
  typedef TContext Context;

  class Dispatcher {
    FORBID_COPY(Dispatcher);
    friend class MPIGrainRunner;

   private:
    class MPIMasterTask : public Task {
     private:
      struct MPIGrainRunner *runner_;

     public:
      MPIMasterTask(MPIGrainRunner *runner_in) {
        runner_ = runner_in;
      }

      void Run() {
        int rank;
        int message;
        int n_slaves_alive = 0;
        int n_slaves_busy = 0;

        DEBUG_MSG(1.0, "DISPATCH: Firing up the cannons.");
        DEBUG_MSG(1.0, "DISPATCH: We will accomplish %u tasks.",
            unsigned(runner_->dispatcher_->queue_->size()));

        for (;;) {
          runner_->RecvInt_(&rank, &message);

          runner_->mutex_.Lock();
          Grain* grain = runner_->dispatcher_->queue_->Pop();
          runner_->mutex_.Unlock();

          if (message == BIRTH) {
            DEBUG_MSG(1.0, "DISPATCH: Received birth message.");
            n_slaves_alive++;
            n_slaves_busy++;
          } else {
            DEBUG_ASSERT(message == GIVE_ME_WORK);
            DEBUG_MSG(1.0, "DISPATCH: Somebody wants more work.");
          }

          if (grain) {
            char buf[sizeof(Grain) + 1];

            buf[0] = 1; // data is available
            mem::CopyBytes(buf + 1, grain, sizeof(*grain));
            delete grain;

            DEBUG_MSG(1.0, "DISPATCH: Sending work on over.");
            MPI_Send(buf, sizeof(buf), MPI_CHAR, rank,
                runner_->tag_, MPI_COMM_WORLD);
          } else if (n_slaves_alive == runner_->n_slaves_) {
            // we have to make sure all workers are born before we quit,
            // otherwise the worker will wait infinitely for work.
            break;
          }
        }

        DEBUG_MSG(1.0, "DISPATCH: Waiting for workers to die...");


        /* Wait for all to die */
        while (n_slaves_alive != 0) {
          // we received a message from the last loop

          if (message == GIVE_ME_WORK) {
            char buf[1];

            buf[0] = 0; // tell them to die
            MPI_Send(buf, sizeof(buf), MPI_CHAR, rank,
                runner_->tag_, MPI_COMM_WORLD);

            DEBUG_MSG(1.0, "DISPATCH: I told a worker to die.");
            n_slaves_busy--;
          } else if (message == DEATH) {
            DEBUG_MSG(1.0, "DISPATCH: A worker died.");
            n_slaves_alive--;
          } else {
            DEBUG_ASSERT_MSG(0, "DISPATCHED: Message was %d??",
                message);
          }

          if (n_slaves_alive != 0) {
            runner_->RecvInt_(&rank, &message);
          }
        }

        DEBUG_MSG(1.0, "DISPATCH: All workers have died.");

        delete this;
      }
    };

   private:
    MPIGrainRunner *runner_;
    GrainQueue<Grain> *queue_;

   public:
    Dispatcher() {}
    ~Dispatcher() {}

    void Init(MPIGrainRunner *runner_in) {
      queue_ = NULL;
      runner_ = runner_in;
    }

    void set_queue(GrainQueue<Grain> *queue_in) {
      queue_ = queue_in;
    }

   private:
    void MasterLoop_() {
      if (runner_->n_slaves_ > 0) {
        MPIMasterTask *task = new MPIMasterTask(runner_);
        task->Run();
      }
    }
  };

 private:
  class ConsumerTask : public Task {
    FORBID_COPY(ConsumerTask);

   private:
    struct MPIGrainRunner *runner_;

   public:
    ConsumerTask(MPIGrainRunner *runner_in) {
      runner_ = runner_in;
    }

    void Run() {
      int my_grains = 0;
      for (;;) {
        Grain *grain = runner_->NextGrain_();
        if (!grain) {
          break;
        }
        grain->Run(runner_->context_);
        delete grain;
        my_grains++;
      }
      DEBUG_MSG(1.0, "A thread on rank %d is done, completed %d grains.",
          runner_->my_rank_, my_grains);
      delete this;
    }
  };

 private:
  enum { BIRTH = 27, GIVE_ME_WORK, DEATH };

 private:
  int tag_;
  Context context_;
  int master_rank_;
  int my_rank_;
  int n_nodes_;
  int n_slaves_;
  Dispatcher *dispatcher_;

  WaitCondition need_work_cond_;
  volatile int need_work_;
  WaitCondition have_work_cond_;
  volatile int have_work_;
  Mutex mutex_;
  ArrayList<Grain *> slave_grains_;

 public:
  MPIGrainRunner() {}
  ~MPIGrainRunner() {}

  /**
   * Initialize this.
   *
   * @param name name to associate information with
   * @param tag_in the tag for this to use; this will occupy both the
   *               specified tag AND the one after it
   * @param context_in the context to run grains with
   * @param master_rank_in the rank of the master (defaults to first)
   */
  void Init(const char *name, int tag_in, Context context_in, int master_rank_in = 0) {
    tag_ = tag_in;
    context_ = context_in;
    master_rank_ = master_rank_in;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodes_);
    n_slaves_ = n_nodes_ - 1;
    DEBUG_MSG(1.0, "my_rank = %d", my_rank_);
    if (master_rank_ == my_rank_) {
      dispatcher_ = new Dispatcher();
      dispatcher_->Init(this);
    } else {
      dispatcher_ = NULL;
    }
    have_work_ = 0;
    need_work_ = 0;
    slave_grains_.Init();
    xrun_subparam_set(name, "n_nodes", "%d", n_nodes_);
  }

  Dispatcher *dispatcher() const {
    return dispatcher_;
  }

  Thread *SpawnThread() {
    ConsumerTask *task = new ConsumerTask(this);
    Thread *thread = new Thread();
    thread->Init(task);
    thread->Start();
    return thread;
  }

  /**
   * Creates the specified number of threads, and uses those to execute
   * all grains of work.
   *
   * Use this rather than the other SpawnThread methods.
   */
  void RunThreads(int num_threads) {
    DEBUG_MSG(1.0, "Rank %d is ready to roll.", my_rank_);
    ArrayList<Thread*> threads;
    int num_worker_threads = num_threads;

    if (num_worker_threads != 2) abort();
    threads.Init(num_worker_threads);

    for (int i = 0; i < num_worker_threads; i++) {
      threads[i] = SpawnThread();
    }
    if (dispatcher_) {
      dispatcher_->MasterLoop_();
      DEBUG_MSG(1.0, "Master loop done, cleaning up.");
    } else {
      SlaveLoop_();
      DEBUG_MSG(1.0, "Slave %d loop done, cleaning up.", my_rank_);
    }
    for (int i = num_worker_threads; i--;) {
      threads[i]->WaitStop();
      delete threads[i];
    }
    if (!dispatcher_) {
      // Let the master know I'm done.
      SendInt_(master_rank_, DEATH);
    }
    DEBUG_MSG(1.0, "Rank %d killed all threads.", my_rank_);
  }

 private:
  void SendInt_(int dest_rank, int num) {
    MPI_Send(&num, 1,
      MPI_INT, dest_rank,
      tag_ + 1,
      MPI_COMM_WORLD);
  }

  void RecvInt_(int *send_rank, int *num_ptr) {
    MPI_Status status;
    MPI_Recv(num_ptr, 1,
      MPI_INT,
      MPI_ANY_SOURCE,
      tag_ + 1,
      MPI_COMM_WORLD,
      &status);
    *send_rank = status.MPI_SOURCE;
  }

  Grain *NextGrain_() {
    Grain *grain;

    if (dispatcher_) {
      DEBUG_ASSERT((my_rank_ == master_rank_));
      mutex_.Lock();
      grain = dispatcher_->queue_->Pop();
      mutex_.Unlock();
    } else {
      DEBUG_ASSERT((my_rank_ != master_rank_));

      mutex_.Lock();
      need_work_++;
      mutex_.Unlock();

      need_work_cond_.Signal();

      mutex_.Lock();
      while (!have_work_) {
        have_work_cond_.Wait(&mutex_);
      }
      if (slave_grains_.size() > 0) {
        grain = *slave_grains_.PopBackPtr();
        have_work_--;
      } else {
        grain = NULL;
      }
      mutex_.Unlock();
      DEBUG_MSG(2.0, "Slave gave me stuff!");
    }

    return grain;
  }

  void SlaveLoop_() {
    bool done = false;

    DEBUG_MSG(1.0, "%d, WORKER: Announcing birth...", my_rank_);
    SendInt_(master_rank_, BIRTH);

    while (!done) {
      char buf[sizeof(Grain) + 1] = "q";
      MPI_Status status;

      DEBUG_MSG(1.0, "%d, WORKER: Waiting for work...", my_rank_);
      MPI_Recv(buf, sizeof(buf),
        MPI_CHAR, MPI_ANY_SOURCE,
        tag_,
        MPI_COMM_WORLD, &status);

      Grain *grain = NULL;

      if (buf[0] == 1) {
        DEBUG_MSG(1.0, "%d, WORKER: Got some work.  There are %d waiting.",
            my_rank_, need_work_);

        grain = new Grain();
        mem::CopyBytes(grain, buf+1, sizeof(Grain));

        mutex_.Lock();
        while (need_work_ == 0) {
          need_work_cond_.Wait(&mutex_);
        }
        need_work_--;
        have_work_++;
        *slave_grains_.AddBack() = grain;
        mutex_.Unlock();

        have_work_cond_.Signal();

        SendInt_(master_rank_, GIVE_ME_WORK);
      } else {
        DEBUG_ASSERT_MSG(buf[0] == 0, "buf[0] = %d", buf[0]);
        grain = NULL;
        done = true;
      }
    }

    DEBUG_MSG(1.0, "%d, WORKER: Duly dying !!!!!!!!!!!!!!!!!!!", my_rank_);

    mutex_.Lock();
    have_work_ = -1;
    mutex_.Unlock();

    have_work_cond_.Broadcast();
  }
};

#endif
