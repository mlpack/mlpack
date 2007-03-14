// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file task.h
 *
 * Declaration for generic task concept.
 */
 
#ifndef PAR_TASK_H
#define PAR_TASK_H

#include "base/cc.h"
#include "base/common.h"

/**
 * Single start-to-finish task to be executed.
 *
 * This is a polymorphic class.
 */
class Task {
  FORBID_COPY(Task);
  
 public:
  Task() {}
  virtual ~Task() {}
  
  virtual void Run() = 0;
};

#endif
