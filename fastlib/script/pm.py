
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
  raise Exception("The strings '%s' and '%s' conflict." % (a, b))

def mybasename(str):
  #def myindex(str, c):
  #  try:
  #    return str.rindex(c) + 1
  #  except:
  #    return 0
  #result = str[max(myindex(str, "/"), myindex(str, "\\")):]
  #return result
  return os.path.basename(str)

class RunSpec:
  """One particular combination of parameters, that corresponds
  to one actual run of the program.
  """
  def __init__(self):
    """Create an empty run specification.
    """
    self.binfile = None
    self.param_args = []
    self.param_info = []
    self.wrapper_args = []
    self.wrapper_info = []
    self.inputs = []
    self.outputs = []
    self.stdin = None
    self.stdout = None
  def merge(self, other):
    """Merge the other's fields into this.
    
    A Combine exception is raised if singleton parameters
    conflict.
    
    Dictionary-based 'Var' parameters are overwritten.
    """
    self.binfile = combine(self.binfile, other.binfile)
    self.param_args += other.param_args
    self.param_info += other.param_info
    self.wrapper_args += other.wrapper_args
    self.wrapper_info += other.wrapper_info
    self.inputs += other.inputs
    self.outputs += other.outputs
    self.stdin = combine(self.stdin, other.stdin)
    self.stdout = combine(self.stdout, other.stdout)
  def merged_with(self, other):
    """Return a copy of this RunSpec merged with the other's
    fields.
    """
    c = copy.deepcopy(self)
    c.merge(other)
    return c
  def check(self):
    assert self.binfile != None
  def generate_name(self, include_executable = True):
    """Generates a user-friendly filename.
    """
    self.check()
    allparas = []
    allparas += self.wrapper_info
    if include_executable:
      allparas += [os.path.basename(self.binfile)]
    allparas += self.param_info
    if self.stdin != None:
      allparas += ["stdin_%s" % os.path.basename(self.stdin)]
    return "__".join(allparas)
  def to_args(self):

    """Turns to a list of arguments such that args[0] is the binary file,
    and the rest are the parameters, suitable for os.exec.
    
    This does not include the stdin or stdout redirects, so make sure you
    handle them separately.
    """
    return self.wrapper_args + [self.binfile] + self.param_args
  def to_command(self):
    """Turns into a single shell command.
    """
    # TODO: Escaping
    allparas = [util.shellquote(x) for x in self.to_args()]
    if self.stdin != None:
      allparas += ["<%s" % util.shellquote(self.stdin)]
    if self.stdout != None:
      allparas += [">%s" % util.shellquote(self.stdout)]
    return " ".join(allparas)
  def get_all_inputs(self):
    result = self.inputs
    if self.stdin:
      result = result + [self.stdin]
    return result
  all_inputs = property(get_all_inputs)

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
      #print e.to_command()
      print e.generate_name() + ": " + e.binfile + " " + " ".join(e.param_args)

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
  def __init__(self, *choices):
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

# This is almost certainly broken -- we've redesigned how to do cross
# validation.
#
#class CrossValidate(ParamSet):
#  """
#  WARNING! HASN'T BEEN TRIED YET
#  
#  Runs several runs over the data set, varying which subset is used for
#  training and testing.
#  
#  NOTE TO SELF: Eventually we would want "optimize over cross-validate"
#  to work properly (and not find the best portion to cross validate over).
#  """
#  def __init__(self, varname_train, varname_test, count, *files):
#    """Creates a cross-validation set of runs.
#    
#    The parameter names of train and test are provided, and it is assumed
#    the program being run understands the :x-2/5 and :x2/5 syntax used to
#    denote cross validation.
#    
#    You then provide the number of ways you want to do cross validation, and
#    also the list of file names, or just a single file name, to run over.
#    
#    Example 1: CrossValidate("train", "test", 10, "a.txt")
#    
#    Example 2: CrossValidate("train_set", "test_set", 20, "a.txt", "b.txt")
#    """
#    self.varname_train = varname_train
#    self.var_train = Input(varname_train)
#    self.varname_test = varname_test
#    self.var_test = Input(varname_test)
#    self.files = files
#    self.count = count
#  def enumerate(self):
#    all = []
#    for file in self.files:
#      for i in range(0, count):
#        spec = RunSpec()
#        self.var_train.set(spec, file)
#        self.var_test.set(spec, file)
#        spec.params[self.varname_train + "/subset"] = ("x-%d/%d" % (i, count))
#        spec.params[self.varname_test + "/subset"] = ("x%d/%d" % (i, count))
#        all.append(spec)
#    return all

# Destinations that can be bound to

class BindDest:
  """Anything that can be bound to, a RunSpec field."""
  def set(self, spec, val):
    """Sets the corresponding field in the RunSpec to the
    given value."""
    pass

class Param(BindDest):
  """Any single parameter argument.
  
  Parameter is formatted in the format:
  
      pre + <value> + post
  
  For instance, if pre = "--length=" and post is "", then the resulting
  string for a value of "12" would be "--length=12".
  """
  def __init__(self, pre, post, show_in_filename = True):
    # prefix and suffix
    self.pre = pre
    self.post = post
    # prefix and suffix for filenames
    pre_base = pre.lstrip("-")
    if "=" in pre_base:
      pos = pre_base.index("=")
      pre_base = pre_base[:pos] + "=" + mybasename(pre_base[pos+1:])
    self.info_pre = util.sanitize_basename(pre_base)
    self.info_post = util.sanitize_basename(post)
    self.show_in_filename = show_in_filename
  def set(self, spec, val):
    val = str(val)
    spec.param_args.append(self.pre + val + self.post)
    if self.show_in_filename:
      val_pretty = util.sanitize_basename(mybasename(val))
      spec.param_info.append(self.info_pre + val_pretty + self.info_post)

class Var(Param):
  """A regular parameter, such as --length.
  """
  def __init__(self, name):
    Param.__init__(self, "--%s=" % name, "")

class Input(Var):
  """A parameter that corresponds to be an input file.
  
  This field is suitable for enforcing file dependencies.
  
  This will also truncate any characters after ':' operator for dependency
  purposes; the ':' is treated special denoting particular kinds of subsets,
  used in cross validation.
  """
  def set(self, spec, val):
    Var.set(self, spec, val)
    spec.inputs += [val]

class Output(Param):
  """A parameter that corresponds to be an output file.
  
  This field is suitable for enforcing file dependencies.
  """
  def __init__(self, name):
    Param.__init__(self, "--%s=" % name, "", False)
  def set(self, spec, val):
    Param.set(self, spec, val)
    spec.outputs += [val]

class Extra(Param):
  """An extra parameter that doesn't directly fit into the
  fx system.
  """
  def __init__(self):
    Param.__init__(self, "", "", True)

class Stdin(BindDest):
  """A parameter that corresponds to be standard input.
  
  This field is suitable for enforcing file dependencies.
  """
  def __init__(self):
    pass
  def set(self, spec, val):
    spec.stdin = str(val)

class Stdout(BindDest):
  """A parameter that corresponds to be standard output.
  
  This field is suitable for enforcing file dependencies.
  """
  def __init__(self):
    pass
  def set(self, spec, val):
    spec.stdout = str(val)

class Binfile(BindDest):
  """The binary file to be executed.
  """
  def set(self, spec, val):
    spec.binfile = str(val)

class MpiCluster(BindDest):
  """A parameter that represents running in MPI.
  """
  def __init__(self, machinefile):
    """Sample use:
      pm.Bind(pm.MpiCluster(os.path.abspath("./amdmachines.txt")), [1, 2, 4, 8, 12])
    """
    self.machinefile = machinefile
  def set(self, spec, val):
    spec.wrapper_args += ["mpirun", "-machinefile", self.machinefile, "-np", str(val)]
    spec.wrapper_info += ["mpi-%s-%s" % (os.path.basename(self.machinefile), str(val))]

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

# Useful functions

def paramset_from_args(arguments):
  """Turns a list of command-line arguments into a parameter set.
  
  For each parameter, the user can specify multiple values.  There are
  two different syntaxes.  If the parameter is in the format --x=y then:
  
     --x=1,2,3   (x has 3 values, 1 2 and 3)
     --x=1,      (just 1 value, but you want this to be included in name)
  
  Note it is still valid to say "--x=1".  However, "--x=1" will not be
  reflected in the filename (it is deemed unimportant); however, if a
  comma is stuck at the end of the line, it will be included.
  
  Similarly, any parameter (not just --x=y) can be reflected:
  
     foo{1,2,3}
     foo{1}
  
  Similarly, foo1 is also valid, but it won't be reflected in the filename
  unless it has braces.
  """
  elements = []

  for arg in arguments:
    if "{" in arg and "}" in arg:
      open_index = arg.index("{")
      close_index = arg.rindex("}")
      choices_str = arg[open_index+1:close_index]
      pre = arg[:open_index]
      post = arg[close_index+1:]
      choices = [x for x in choices_str.split(",") if x != ""]
      element = Bind(Param(pre, post, True), choices)
    elif arg.startswith("--") and "=" in arg and "," in arg:
      eq_index = arg.index("=")
      choices_str = arg[eq_index+1:].strip(",")
      varname = arg[2:eq_index]
      choices = [x for x in choices_str.split(",") if x != ""]
      element = Bind(Var(varname), choices)
    else:
      element = Bind(Param("", "", False), arg)
    elements += [element]
    
  return Combine(*elements)

# Tests

class ParamTest(unittest.TestCase):
  def setUp(self):
    self.params = Combine(
      Bind(Binfile(), Val("/usr/bin/sort")),
      Bind(Input("infile"), Vals("/foo/bar/in1.txt", "/foo/bar/in2.txt")),
      Bind(Output("outfile"), Vals("/foo/baz/out1.txt")),
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
      return len([x for x in all if ("--%s=%s" % (param, str(val))) in x.param_args])
    self.assertEqual(15, count("infile", "/foo/bar/in1.txt"))
    self.assertEqual(10, count("useless", 1.3))

  def test_names(self):
    all = self.params.enumerate()
    names = [x.generate_name() for x in all]
    self.assert_("sort__infile_in1.txt__useless_1.1__bw1_1__bw2_0" in names)

  def test_getters_setters(self):
    all = self.params.enumerate()
    first = all[0]
    self.assertEqual(first.all_inputs, ["/foo/bar/in1.txt"])
    first.stdin = "foostdin"
    self.assertEqual(first.all_inputs, ["/foo/bar/in1.txt", "foostdin"])

  def test_order(self):
    all = self.params.enumerate()
    names = [x.generate_name() for x in all]
    self.assertEqual(names[0], "sort__infile_in1.txt__useless_1.1__bw1_1__bw2_0")
    self.assertEqual(names[1], "sort__infile_in1.txt__useless_1.1__bw1_2__bw2_1")
    self.assertEqual(names[2], "sort__infile_in1.txt__useless_1.1__bw1_4__bw2_2")
    self.assertEqual(names[5], "sort__infile_in1.txt__useless_1.3__bw1_1__bw2_0")
    self.assertEqual(names[15], "sort__infile_in2.txt__useless_1.1__bw1_1__bw2_0")
  
  def test_mybasename(self):
    self.assertEqual("foo.txt", mybasename("/bak/foo.txt"))
    self.assertEqual("foo.txt", mybasename("./bak/foo.txt"))
    self.assertEqual("foo.txt", mybasename("bak/foo.txt"))
    self.assertEqual("foo.txt", mybasename("bak/oaisjd/asdlfasl/foo.txt"))
    self.assertEqual("abs.txt", mybasename("bak/oaisjd/asdlfasl" + os.sep + "abs.txt"))

if __name__ == "__main__":
  unittest.main()
