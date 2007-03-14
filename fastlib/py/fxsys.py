"""FASTexec system management.

Allows you to create, manage, and deploy experiments.
"""

# work items:
#  - path translation
#  - argument translation (huh?)

import os
import random
import util
import pm
import sys
import unittest
import work

class FxSystemException:
  pass

class FxSystem:
  """A particular fx system, parameterized by the root of the hierarchy.
  
  Important attributes:
  - rootpath: the root of the system
  """

  def __init__(self, rootpath):
    """Creates a new fx-system based on the specified path.
    
    Rootpath must exist.  If the proper FASTexec subdirectories are not present
    they will be created, but the root itself must be created.  You may
    consider making sure the root directory itself has the group
    setguid bit if multiple people use the same system.
    
    - rootpath: absolute path of the root of the fx system
    """
    self.rootpath = rootpath
    try:
      self.validate()
    except:
      self.create()
  
  def validate(self):
    """Makes sure this is a valid fx system.
    """
    if not os.access(os.path.join(self.rootpath, "FXSYS"), os.W_OK):
      raise FxSystemException()
    # TODO(garryb): Include real validation
  
  def create(self):
    """Sets up this path as an fx system.
    
    The rootpath must exit, and must already have permissions set up properly.
    
    If the system is already set up, raises an error exception.
    """
    # TODO(garryb): Implement
    # TODO(garryb): Set permissions properly
    #util.ensuredir(self.rootpath)
    util.ensuredir(os.path.join(self.rootpath, "problems"))
    util.ensuredir(os.path.join(self.rootpath, "data"))
    util.writefile(os.path.join(self.rootpath, "FXSYS"), "FXSYS")
    
    # TODO(garryb):
    
    self.validate()

  def translate(self, relpath):
    """
    Translates a path alias into absolute path.
    
    In particular, paths beginning with FXSYS/ are translated to real paths.
    """
    relpath = str(relpath)
    if relpath.startswith('FXSYS/'):
      result = os.path.join(self.rootpath, relpath[len('FXSYS/'):])
    else:
      result = os.path.abspath(relpath)
    return result

  # All pathname hardcoding should exist in this class, even if it doesn't
  # seem the best place to put it.
  
  def pathof_problem(self, probname):
    return self.translate("FXSYS/problems/%s" % probname)

  def pathof_experiment(self, probname, expname):
    return self.translate("FXSYS/problems/%s/exps/%s" % (probname, expname))

  def pathof_run(self, probname, expname, runname):
    return self.translate("FXSYS/problems/%s/exps/%s/runs/%s" % (probname, expname, runname))

  def get_experiment_names(self, probname):
    return os.listdir(self.translate("FXSYS/problems/%s/exps" % probname))

class Experiment:
  """One particular experiment.
  
  An experiment is a labelled set of runs for a particular purpose.
  """
  
  def __init__(self, probname, expname, create = True, fxsys = None):
    if fxsys == None:
      fxsys = default_fxsys()
    self.fxsys = fxsys
    self.expname = expname
    self.probname = probname
    self.path = fxsys.pathof_experiment(probname, expname)
    if create:
      self.create()
  
  def pathof_run(self, runname):
    return self.fxsys.pathof_run(self.probname, self.expname, runname)
  
  def pathof_statusfile(self, runname):
    fullpath = self.pathof_run(runname)
    return os.path.join(fullpath, work.WorkEntry.STATUSFILE)
  
  def create(self):
    util.ensuredir(self.path)
  
  def execute(self, paramset):
    wq = work.WorkQueue(self)
    wq.add([work.WorkEntry(self, x) for x in paramset.enumerate()])
    wq.inline_exec()
  
  def get_statuses(self):
    """Gets the status, such as work.WorkEntry.FINISHED, for each run that
    has been started.  It is a key-value map of run name to its execution
    status.
    
    If the status file does not exist, is invalid, or cannot be accessed,
    None is returned as the status.
    """
    # This should go through and remove decaying STATUS files
    statuses = {}
    for runname in os.listdir(os.path.join(self.path, "runs")):
      try:
        lines = util.readlines(self.pathof_statusfile(runname))
        statuses[runname] = lines[0].strip()
      except:
        statuses[runname] = None
    return statuses
  
  def cleanup(self, purge = False, nuke = False):
    """Removes the status files of runs that have not finished.
    This is useful you terminate a run and want to restart it -- if you
    do not clean it up, the runs that were in progress will not be restarted.
    
    Be warned -- If you run this while one of the runs is actually
    executing, that run will be re-run in parallel.
    
    purge: whether to completely delete the entire run's directory
           (not just the status file)
    """
    statuses = self.get_statuses()
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
            print "... Status file was already invalid.  This run can be re-run."
      elif nuke:
        print "NUKING COMPLETED RUN %s" % runname
        try:
          util.remove_dir_recursive(self.pathof_run(runname))
        except OSError:
          print "XXX Error deleting the run"
  
  def print_status(self):
    """Prints the log files of any run that is not completed."""
    num_printed = 0
    items = self.get_statuses().items()
    items_ordered = []
    items_ordered += [(x,y) for (x,y) in items if y == work.WorkEntry.FINISHED]
    items_ordered += [(x,y) for (x,y) in items if y != work.WorkEntry.FINISHED]
    for (runname, status) in items_ordered:
      print "%s -- %s" % (status, runname)
      if status != work.WorkEntry.FINISHED:
        logfile = os.path.join(self.pathof_run(runname), work.WorkEntry.LOGFILE)
        try:
          num_printed += 1
          lines = util.readlines(logfile)
          for line in lines:
            print "  | %s" % line.strip()
        except:
          print "  X No log file"
    return num_printed

  def purge(self):
    util.remove_dir_recursive(self.path)

# functions

class NoConfigFileException:
  pass

def default_fxsys():
  """Gets the default Xsys object, by searching for fx.conf in the current
  or parent directories.
  """
  
  # TODO: Cache this
  path = os.getcwd()
  
  while len(path) > 1 and not os.path.exists(os.path.join(path, "fx.conf")):
    path = os.path.dirname(path)

  if len(path) <= 1:
    raise NoConfigFileException
  
  conffile = os.path.join(path, "fx.conf")
  lines = util.readlines(conffile)
  rootpath = os.path.join(path, lines[0])
  return FxSystem(rootpath)

# public test helpers

def test_new_testing_fxsys():
  (tmpdir, tmpfile) = os.path.split(os.tempnam())
  rootpath = "%s/fxsys/%s" % (tmpdir, tmpfile)
  util.ensuredir(rootpath)
  fxsys = FxSystem(rootpath)
  print "Rooted at %s" % rootpath
  fxsys.create()
  return fxsys

def test_create_numlist_infiles(fxsys):
  infiles = []
  for i in [ 2000, 4000, 6000, 8000, 10000 ]:
    fname = fxsys.translate("FXSYS/data/num%d.in" % i)
    util.write_random_ints(fname, i)
    infiles.append(fname)
  return (infiles)

class TestFx(unittest.TestCase):
  
  def setUp(self):
    self.fxsys = test_new_testing_fxsys()
    self.infiles = test_create_numlist_infiles(self.fxsys)
    self.paramset = pm.Combine(
      pm.Bind(pm.Binfile(), pm.Val("/usr/bin/sort")),
      pm.Bind(pm.Extra(), pm.Val("-n")),
      pm.Bind(pm.Stdin(), pm.Vals(*self.infiles)),
      pm.Bind(pm.Stdout(), "out.txt"))
  
  def tearDown(self):
    util.remove_dir_recursive(self.fxsys.rootpath)
  
  def test_inline(self):
    self.exp = Experiment("xtest-sort", "sort-test_inline", fxsys = self.fxsys)
    self.exp.create()
    wq = work.WorkQueue(self.exp)
    wq.add([work.WorkEntry(self.exp, runspec) for runspec in self.paramset.enumerate()])
    print "Inline first"
    wq.inline_exec()
    print "Inline second"
    wq.inline_exec()
    print "Inline end"
  
  def test_shell(self):
    self.exp = Experiment("xtest-sort", "sort-test_shell", fxsys = self.fxsys)
    self.exp.create()
    wq = work.WorkQueue(self.exp)
    wq.add([work.WorkEntry(self.exp, runspec) for runspec in self.paramset.enumerate()])
    print "Writing work queue..."
    fname = wq.write()
    print "Running %s..." % fname
    os.spawnv(os.P_WAIT, "/bin/bash", ["/bin/bash", fname])
    print "Running a second time..."
    os.spawnv(os.P_WAIT, "/bin/bash", ["/bin/bash", fname])
  
  def test_default(self):
    self.exp = Experiment("xtest-sort", "sort-test_default", fxsys = self.fxsys)
    self.exp.execute(self.paramset)

if __name__ == "__main__":
  unittest.main()

