
import copy
import random
import util
import os
import unittest

def combine(a, b):
  if a == None:
    return b
  if b == None:
    return a
  raise Conflict

class Conflict:
  """Exception thrown when two single fields conflict.
  
  For example, perhaps a Combine is done, but the Binfile
  is specified on both sides.
  """
  pass

class RunSpec:
  """One particular combination of parameters, that corresponds
  to one actual run of the program.
  """
  def __init__(self):
    """Create an empty run specification.
    """
    self.binfile = None
    self.wrappers = []
    self.wrapper_info = []
    self.params = {}
    self.extra = []
    self.stdin = None
    self.stdout = None
    self.inputs = []
    self.outputs = []
  def merge(self, other):
    """Merge the other's fields into this.
    
    A Combine exception is raised if singleton parameters
    conflict.
    
    Dictionary-based 'Var' parameters are overwritten.
    """
    self.binfile = combine(self.binfile, other.binfile)
    self.params.update(other.params)
    self.wrappers += other.wrappers
    self.wrapper_info += other.wrapper_info
    self.extra += other.extra
    self.stdin = combine(self.stdin, other.stdin)
    self.stdout = combine(self.stdout, other.stdout)
    self.inputs += other.inputs
    self.outputs += other.outputs
  def merged_with(self, other):
    """Return a copy of this RunSpec merged with the other's
    fields.
    """
    c = copy.deepcopy(self)
    c.merge(other)
    return c
  def check(self):
    assert self.binfile != None
  def generate_name(self):
    """Generates a user-friendly filename.
    """
    self.check()
    allparas = []
    allparas += self.wrapper_info
    allparas += [os.path.basename(self.binfile)]
    allparas += self.extra
    nvp = [pair for pair in self.params.iteritems()]
    nvp.sort() # Sort by parameter name to ensure deterministic ordering
    for (name, val) in nvp:
      if val in self.outputs:
        # output files are not important
        continue
      if val in self.inputs:
        val = "%s_%x" % (os.path.basename(val), hash(val) % 65536)
      allparas.append("%s=%s" % (name, val))
    if self.stdin != None:
      allparas += ["stdin=%s" % os.path.basename(self.stdin)]
    return util.sanitize_basename("_".join(allparas))
  def to_args(self):
    """Turns to a list of arguments such that args[0] is the binary file,
    and the rest are the parameters, suitable for os.exec.
    
    This does not include the stdin or stdout redirects, so make sure you
    handle them separately.
    """
    allparas = self.wrappers + [self.binfile] + self.extra
    allparas += ["--%s=%s" % (k, v) for (k, v) in self.params.iteritems()]
    return allparas
  def to_command(self):
    """Turns into a single shell command.
    """
    # TODO: Escaping
    allparas = []
    allparas += [ util.shellquote(x) for x in self.wrappers ]
    allparas += [util.shellquote(self.binfile)]
    allparas += [ util.shellquote(x) for x in self.extra ]
    allparas += ["--%s=%s" % (k, util.shellquote(v)) for (k, v) in self.params.iteritems()]
    if self.stdin != None:
      allparas += ["<%s" % util.shellquote(self.stdin)]
    if self.stdout != None:
      allparas += [">%s" % util.shellquote(self.stdout)]
    return " ".join(allparas)

# Parameter sets -- sets of all possible run parameters

class ParamSet:
  """Abstract set of parameters.
  """
  def enumerate(self):
    """Returns a list of RunSpecs for all runs that should exist."""
    return []
  def print_all(self):
    """Prints all commands that would be executed if all were to be
    run."""
    for e in self.enumerate():
      print e.to_command()

class Combine(ParamSet):
  """Cartesian product, or all combinations,
  of several smaller parameter sets.
  
  (Technically, this is closer to intersction, but the
  actual 'intersection' operation is not performed for
  redundant parameters.  The handling of redundant
  parameters is undefined.)
  
  Parameters are multiplied so that the last parameters
  vary closest together, and the first in the list
  vary last.  That is, (1 2) x (A B) =
  
     1 A
     1 B
     2 A
     2 B
  """
  def __init__(self, *factors):
    self.factors = factors
  def enumerate(self):
    all = [ RunSpec() ]
    for item in self.factors:
      newlist = []
      enumerated = item.enumerate()
      for runspec1 in all:
        for runspec2 in enumerated:
          newlist.append(runspec1.merged_with(runspec2))
      all = newlist
    return all

class Any(ParamSet):
  """The union of several smaller parameter sets.
  """
  def __init__(self, choices):
    self.choices = choices
  def enumerate(self):
    all = []
    for choice in self.choices:
      all += choice.enumerate()
    return all

class Bind(ParamSet):
  """Binds a particular RunSpec field to a particular
  value set.
  """
  def __init__(self, var, valset):
    self.var = var
    if not isinstance(valset, ValSet):
      if isinstance(valset, list):
        valset = Vals(*valset)
      else:
        valset = Val(valset)
    self.valset = valset
  def enumerate(self):
    all = []
    for val in self.valset.enumerate():
      spec = RunSpec()
      self.var.set(spec, val)
      all.append(spec)
    return all

class CoBind(ParamSet):
  """Binds any number of RunSpec fields to any number
  of value sets, one value set per fields.
  
  When enumerated, the enumerations of every value set are zipped together;
  all enumerations must be of equal size.
  """
  def __init__(self, *pairs):
    self.vars = []
    self.valsets = []
    for i in range(0, len(pairs), 2):
      self.vars.append(pairs[i])
      self.valsets.append(pairs[i + 1])
    # TODO: Runtime check to make sure each set is the same size
  def enumerate(self):
    all = []
    valmatrix = []
    for valset in self.valsets:
      valmatrix.append(valset.enumerate())
    assert min(map(len, valmatrix)) == max(map(len, valmatrix))
    for i in range(len(valmatrix[0])):
      spec = RunSpec()
      for j in range(len(valmatrix)):
        self.vars[j].set(spec, valmatrix[j][i])
      all.append(spec)
    return all

class CrossValidate(ParamSet):
  """
  WARNING! HASN'T BEEN TRIED YET
  
  Runs several runs over the data set, varying which subset is used for
  training and testing.
  
  NOTE TO SELF: Eventually we would want "optimize over cross-validate"
  to work properly (and not find the best portion to cross validate over).
  """
  def __init__(self, varname_train, varname_test, count, *files):
    """Creates a cross-validation set of runs.
    
    The parameter names of train and test are provided, and it is assumed
    the program being run understands the :x-2/5 and :x2/5 syntax used to
    denote cross validation.
    
    You then provide the number of ways you want to do cross validation, and
    also the list of file names, or just a single file name, to run over.
    
    Example 1: CrossValidate("train", "test", 10, "a.txt")
    
    Example 2: CrossValidate("train_set", "test_set", 20, "a.txt", "b.txt")
    """
    self.varname_train = varname_train
    self.var_train = Input(varname_train)
    self.varname_test = varname_test
    self.var_test = Input(varname_test)
    self.files = files
    self.count = count
  def enumerate(self):
    all = []
    for file in self.files:
      for i in range(0, count):
        spec = RunSpec()
        self.var_train.set(spec, file)
        self.var_test.set(spec, file)
        spec.params[self.varname_train + "/subset"] = ("x-%d/%d" % (i, count))
        spec.params[self.varname_test + "/subset"] = ("x%d/%d" % (i, count))
        all.append(spec)
    return all

# Destinations that can be bound to

class BindDest:
  """Anything that can be bound to, a RunSpec field."""
  def set(self, spec, val):
    """Sets the corresponding field in the RunSpec to the
    given value."""
    pass

class Var(BindDest):
  """A regular parameter, such as --length.
  """
  def __init__(self, name):
    self.name = name
  def set(self, spec, val):
    spec.params[self.name] = val

class MpiCluster(BindDest):
  """A parameter that represents running in MPI.
  """
  def __init__(self, machinefile):
    """Sample use:
      pm.Bind(pm.MpiCluster(os.path.abspath("./amdmachines.txt")), [1, 2, 4, 8, 12])
    """
    self.machinefile = machinefile
  def set(self, spec, val):
    spec.wrappers += ["mpirun", "-machinefile", self.machinefile, "-np", str(val)]
    spec.wrapper_info += ["mpi-%s-%s" % (os.path.basename(self.machinefile), str(val))]

class Input(Var):
  """A parameter that corresponds to be an input file.
  
  This field is suitable for enforcing file dependencies.
  
  This will also truncate any characters after ':' operator for dependency
  purposes; the ':' is treated special denoting particular kinds of subsets,
  used in cross validation.
  """
  def set(self, spec, val):
    spec.inputs.append(val)
    Var.set(self, spec, val)

class Output(Var):
  """A parameter that corresponds to be an output file.
  
  This field is suitable for enforcing file dependencies.
  """
  def set(self, spec, val):
    spec.outputs.append(str(val))
    Var.set(self, spec, val)

class Extra(BindDest):
  """An extra parameter that doesn't directly fit into the
  fx system.
  """
  def set(self, spec, val):
    spec.extra.append(val)

class Stdin(BindDest):
  """A parameter that corresponds to be standard input.
  
  This field is suitable for enforcing file dependencies.
  """
  def set(self, spec, val):
    spec.stdin = str(val)

class Stdout(BindDest):
  """A parameter that corresponds to be standard output.
  
  This field is suitable for enforcing file dependencies.
  """
  def set(self, spec, val):
    spec.stdout = str(val)

class Binfile(BindDest):
  """The binary file to be executed.
  """
  def set(self, spec, val):
    spec.binfile = str(val)

# Set of values that can be bound to a destination

class ValSet:
  """An arbitrary set of values that may be bound to a variable.
  """
  def enumerate(self):
    return []

class Vals(ValSet):
  """A pre-specified enumeration of values.
  """
  def __init__(self, *vals):
    self.vals = vals
  def enumerate(self):
    return self.vals

class Val(Vals):
  """A pre-specified single value.
  """
  def __init__(self, val):
    Vals.__init__(self, val)

# Tests

class ParamTest(unittest.TestCase):
  def setUp(self):
    self.params = Combine(
      Bind(Binfile(), Val("/usr/bin/sort")),
      Bind(Input("infile"), Vals("in1.txt", "in2.txt")),
      Bind(Var("useless"), Vals(1.1, 1.3, 1.7)),
      CoBind(Var("bw1"), Vals(1, 2, 4, 8, 16), Var("bw2"), Vals(0, 1, 2, 3, 4)))
  
  def test_print(self):
    # TODO - Doesn't test anything
    self.params.print_all()
  
  def test_len(self):
    self.assertEqual(30, len(self.params.enumerate()))

  def test_product(self):
    all = self.params.enumerate()
    def count(param, val):
      return len([x for x in all if x.params[param] == val])
    self.assertEqual(15, count("infile", "in1.txt"))
    self.assertEqual(10, count("useless", 1.3))

if __name__ == "__main__":
  unittest.main()
