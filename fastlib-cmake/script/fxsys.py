"""FASTexec experiment system management.

Allows you to create, manage, and deploy experiments.
"""

import util
import work

import os
import sys

class AbstractExperiment:
  """Abstract "experiment" according to the experiment system.
  
  The important things an experiment does:
   - figures out where to store results on the filesystem
   - helps user manage partially completed runs
  """
  def __init__(self, path):
    """Initializes an experiment rooted at the specified directory."""
    self.path = path
  
  def get_status_of_attempted_runs(self):
    """Gets the status, such as work.WorkEntry.FINISHED, for each run that
    has been started.  It is a key-value map of run name to its execution
    status.

    If the status file does not exist, is invalid, or cannot be accessed,
    None is returned as the status.
    """
    # This should go through and remove decaying STATUS files
    statuses = {}
    for runname in os.listdir(self.pathof_runs()):
      try:
        lines = util.readlines(self.pathof_statusfile(runname))
        statuses[runname] = lines[0].strip()
      except:
        statuses[runname] = None
    return statuses

  def generate_name(self, runspec):
    """Generates a name for a pm.RunSpec."""
    raise Exception("Not implemented")

  def pathof_runs(self):
    """Returns the path where all runs are stored as directories.
    
    For example, an ls of this directory will return all relevant runs."""
    return self.path
  
  def pathof_run(self, runname):
    """Returns the path of a particular run, given its name."""
    return os.path.join(self.path, runname)
  
  def pathof_statusfile(self, runname):
    """Returns the path of the status file for a run."""
    return os.path.join(self.pathof_run(runname), work.WorkEntry.STATUSFILE)

  def execute(self, paramset):
    """Runs a ParamSet under this "experiment"."""
    wq = work.WorkQueue(self)
    wq.add([work.WorkEntry(self, x) for x in paramset.enumerate()])
    wq.inline_exec()

  def print_status(self, detail_finished = False, show_finished = True):
    """Prints the log files of any run that is not completed."""
    num_printed = 0
    items = self.get_status_of_attempted_runs().items()
    items_ordered = []
    if show_finished:
      items_ordered += [(x,y) for (x,y) in items if y == work.WorkEntry.FINISHED]
    items_ordered += [(x,y) for (x,y) in items if y != work.WorkEntry.FINISHED]
    for (runname, status) in items_ordered:
      print status, "--", runname
      if status != work.WorkEntry.FINISHED or detail_finished:
        logfile = os.path.join(self.pathof_run(runname), work.WorkEntry.LOGFILE)
        try:
          num_printed += 1
          lines = util.readlines(logfile)
          if len(lines) == 0:
            print "  (Log file is empty)"
          else:
            for line in lines:
              print "  | %s" % line.strip()
        except:
          print "  X No log file"
    return num_printed

  def cleanup(self, purge = False, nuke = False):
    """Cleans up partially completed runs.
    
    purge - if this is false, only the status file of partically complete
    runs is removed, otherwise, the entire directory is removed
    
    nuke - remove fully completed runs too"""
    statuses = self.get_status_of_attempted_runs()
    for (runname, status) in statuses.items():
      if status != work.WorkEntry.FINISHED:
        statusfile = self.pathof_statusfile(runname)
        if purge or nuke:
          print "Purging stale run %s" % runname
          try:
            util.remove_dir_recursive(self.pathof_run(runname))
          except OSError:
            print "XXX Error deleting the run"
        else:
          print "Clearing status file of stale run %s" % runname
          try:
            os.unlink(statusfile)
          except:
            print "... Status file was already cleared.  This run can be re-run."
      elif nuke:
        print "NUKING COMPLETED RUN %s" % runname
        try:
          util.remove_dir_recursive(self.pathof_run(runname))
        except OSError:
          print "XXX Error deleting the run"

class SimpleExperiment(AbstractExperiment):
  """Experiment that takes existence in the current directory.
  """
  def __init__(self, current_path, exename, label):
    """Initializes:
    
    - current_path: the current directory (an "fx" directory will be
      created here)
    - exename: name of executable
    - label: a label given to this experiment
    
    Each run goes in a directory of form:
    
      current_path/fx/exename/label/runname
    """
    path = os.path.join(current_path, "fx", os.path.basename(exename), label)
    AbstractExperiment.__init__(self, path)
    self.exename = exename
    self.label = label

  def generate_name(self, runspec):
    """Generates a name for a pm.RunSpec."""
    # generate a name WITHOUT the executable file in the name
    return runspec.generate_name(False)

def execute_paramset_here(exename, label, paramset, current_path = "."):
  """Executes a parameter set in the current directory.
  """
  experiment = SimpleExperiment(os.path.abspath(current_path),
      os.path.basename(exename), label)
  experiment.execute(paramset)

# Name of the output file for FX data to use by default.
OUTPUT_FNAME = "output.txt"
