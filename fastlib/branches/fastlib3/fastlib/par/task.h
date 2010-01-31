// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file task.h
 *
 * Declaration for generic task concept.
 */
 
#ifndef PAR_TASK_H
#define PAR_TASK_H

#include "fastlib/base/base.h"

/**
 * Single start-to-finish task to be executed.
 *
 * This is a polymorphic class.
 */
class Task {
  FORBID_ACCIDENTAL_COPIES(Task);
  
 public:
  Task() {}
  virtual ~Task() {}
  
  virtual void Run() = 0;
};

#endif
