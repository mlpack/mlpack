import os
import datastore
import util

# TODO: Who translates?

class WorkEntry:
  """A single entry on the work queue.
  
  Important attributes:
  - binfile: the binary file to execute
  - params: parameters passed, as a raw list
  - inputs: all files required by this stage
  - outputs: all files written by this stage
  """
  
  INPUT_NOT_AVAILABLE = "input_not_available"
  IN_PROGRESS = "in_progress"
  FINISHED = "finished"
  ERROR = "error"
  STATES = [INPUT_NOT_AVAILABLE, IN_PROGRESS, FINISHED, ERROR]
  
  STATUSFILE = "STATUS"
  LOGFILE = "LOG"
  
  def __init__(self, experiment, runspec):
    """Creates a new work queue entry.
    """
    self.experiment = experiment
    self.runname = runspec.generate_name()
    self.rundir = experiment.pathof_run(self.runname)
    self.runspec = runspec
    self.logfile = os.path.join(self.rundir, WorkEntry.LOGFILE)
    self.statusfile = os.path.join(self.rundir, WorkEntry.STATUSFILE)
  
  def to_datastore(self, node):
    #TODO: FINISH
    node.get_subnode_create()
    self.fxsys.to_datastore(node)
    node.set_val_path("/rundir", self.rundir)
    # TODO: serialize the runspec
  
  def to_bash(self):
    conditions = []
    conditions.append("! -e %s" % (util.shellquote(self.statusfile)))
    for input in self.runspec.inputs:
      conditions.append("-e %s" % util.shellquote(input))
    if self.runspec.stdin != None:
      conditions.append("-e %s" % util.shellquote(self.runspec.stdin))
    command = self.runspec.to_command()
    id = util.shellquote(os.path.basename(self.runname))
    lines = [
        "echo %s" % id,
        "if %s" % " && ".join([ "[ %s ]" % x for x in conditions]),
        "then echo ' ... Starting'",
        "mkdir -p %s" % util.shellquote(self.rundir),
        "cd %s" % util.shellquote(self.rundir),
        "echo '%s' >%s" % (WorkEntry.IN_PROGRESS, WorkEntry.STATUSFILE),
        "%s 2>%s" % (command, util.shellquote(self.logfile)),
        "RV=$?",
        "if [ \"$RV\" == 0 ]",
        "then echo '%s' >%s" % (WorkEntry.FINISHED, WorkEntry.STATUSFILE),
        "echo ' ... Done!'",
        "else echo '%s' >%s" % (WorkEntry.ERROR, WorkEntry.STATUSFILE),
        "echo ' ... Error' $RV",
        "fi",
        "fi"
    ]
    return "; ".join(lines)
  
  def inline_exec(self):
    # TODO: Recover from errors
    print "%s" % self.runname
    try:
      for infile in self.runspec.inputs:
        if not os.path.exists(infile):
          return self.INPUT_NOT_AVAILABLE
      if not os.path.exists(self.rundir):
        util.ensuredir(self.rundir)
    except OSError:
      print "ERROR: Error reading input files: %s" % (str(self.runspec.inputs))
      return self.ERROR
    try:
      if not util.createlock(self.statusfile):
        try:
          lines = util.readlines(self.statusfile)
          WorkEntry.STATES.index(lines[0]) # raise exception if not a valid state
          return lines[0]
        except OSError:
          print "ERROR: Could not read status file %s." % self.statusfile
          return self.ERROR
      else:
        print " ... %s" % self.runspec.to_command()
        util.writefile(self.statusfile, self.IN_PROGRESS)
        retval = util.spawn_redirect(
          self.rundir, self.runspec.to_args(),
          self.runspec.stdin, self.runspec.stdout, self.logfile)
        if retval != 0:
          status = self.ERROR
          print " ... Error %d " % retval
          print util.readfile(self.logfile)
        else:
          status = self.FINISHED
          print " ... Done!"
        util.writefile(self.statusfile, status)
        return status
    except WorkQueue:
      try:
        util.writefile(self.statusfile, self.ERROR)
      except:
        pass
      return self.ERROR

class WorkQueue:
  """A list of work that needs to be done.
  
  This currently requires to be added in a dependency-friendly order.
  """
  
  def __init__(self, experiment):
    """Creates an empty WorkQueue for the specified system.
    """
    
    self.experiment = experiment
    self.path = experiment.path
    self.entries = []
  
  def add(self, entries):
    self.entries.extend(entries)
  
  def reorder(self):
    # sort by inputs and outputs
    pass
  
  def to_bash_lines(self):
    return ["#!/bin/bash"] + [x.to_bash() for x in self.entries]
  
  def write(self):
    fname = os.path.join(self.path, "workqueue.sh")
    f = open(fname, "a")
    for line in self.to_bash_lines():
      f.write(line)
      f.write("\n")
    f.close()
    return fname
  
  def inline_exec(self):
    # TODO: Handle the return value
    for item in self.entries:
      result = item.inline_exec()
      print result

