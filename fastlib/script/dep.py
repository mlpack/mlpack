# Simple abstract dependency system suitable for generating things like
# makefiles.

import util

import os

class DestFile:
  def __init__(self, name, simplename, dep_files={}, parameterization={}):
    # TODO: Separate "my" parameterization from "dep" parameterization
    self.name = name
    if simplename:
      self.simplename = simplename
    else:
      self.simplename = name
    self.parameterization = parameterization
    self.dep_files = dep_files
  def __repr__(self):
    return self.name
  def __hash__(self):
    return hash(self.name) ^ hash(self.simplename)
  def __cmp__(self, other):
    if isinstance(other, DestFile):
      return cmp(self.name, other.name)
    else:
      return NotImplemented

class DepState:
  def __init__(self, files, params):
    self.files = files
    self.params = params
    self.parameterization = {}
    for file in filemap_to_files(files):
      self.parameterization.update(file.parameterization)

class DepSys:
  def __init__(self):
    self.runs = {}
  def begin(self, state):
    raise Exception("DepSys not overridden")

class DepSysEntry:
  """Abstract dependency system entry.
  
  You must override this and provide a way of generating filenames.
  """
  def __init__(self, sys, state):
    """
    sys - a DepSys object
    state - a DepState object
    """
    self.sys = sys
    self.state = state
  def file(self, simplename, *extra_parameterization):
    parameterization = dict(self.state.parameterization)
    my_parameterization = dict([(k, self.state.params[k])
        for k in extra_parameterization])
    parameterization.update(my_parameterization)
    realname = self._make_name(simplename, parameterization)
    self.ensure_writable(realname)
    return DestFile(realname, simplename, self.state.files, parameterization)
  def _make_name(self, simplename, parameterization):
    raise Exception("not implemented")
  def ensure_writable(realname):
    raise Exception("not implemented")
  def end(self, files):
    raise Exception("not implemented")
  def command(self, str):
    raise Exception("not implemented")

class BadResult:
  pass

class FileCollection:
  def __init__(self, pairs):
    self.files_by_type = {}
    seen_pairs = {}
    for (typename, file) in pairs:
      if not (typename, file) in seen_pairs:
        seen_pairs[(typename, file)] = 1
        self.files_by_type.setdefault(typename, []).append(file)
  def single(self, typename):
    result = self.files_by_type[typename]
    if len(result) != 1:
      raise BadResult()
    return result[0]
  def many(self, typename):
    return self.files_by_type.get(typename, [])
  def to_pairs(self):
    pairs = []
    for (classname, files) in self.files_by_type.items():
      pairs += [(classname, file) for file in files]
    return pairs
  def to_files(self):
    return util.collapse_once(
        [files for (classname, files) in self.files_by_type.items()])
  def to_names(self):
    return [f.name for f in self.to_files()]

def filemap_to_files(filemap):
  return util.collapse_once(
      [file_collection.to_files() for file_collection in filemap.values()])

class Rule:
  """
  Generic rule.
  
  This rule takes in several labelled sets of other rules.
  
  TODO: Explain the idea of "meta-rule" better
  
  This is more a meta-rule, which can depend on other meta-rules.
  Meta-rules are asked for the exact filenames they produce, which
  recursively ask their dependency rules what files they produce.  Once a
  meta-rule knows what files its dependency rules produce, it can generate
  commands to produce its own result files, and tell ITS dependents what
  files it generated.
  
  """
  
  # TODO: (requires thought) If multiple rules depend on a single rule, I
  # *think* this code, how it is written it will run "doit" for that rule
  # multiple times, rather than once.  This might be necessary due to
  # parameterization, I haven't thought much about it.
  
  def __init__(self, **dep_map):
    self.dep_map = dep_map
  def generate(self, sys, params):
    def fake_cont(x):
      pass
    self.run(sys, params, fake_cont)
  def run(self, sys, params, continuation):
    key = (self, util.dicthash(params))
    if key in sys.runs:
      continuation(sys.runs[key])
    else:
      all_deps = []
      
      for (dep_class, deps) in self.dep_map.items():
        for dep in deps:
          all_deps.append((dep_class, dep))
      
      dep_count = [len(all_deps)]
      dep_files = {}
      
      def check_done():
        if dep_count[0] == 0:
          dep_files_converted = dict(self.dep_map)
          for k in self.dep_map:
            dep_files_converted[k] = FileCollection(dep_files.get(k, []))
          state = DepState(dep_files_converted, params)
          # Also memoize here!
          action = sys.begin(state)
          my_files = self.doit(action, dep_files_converted, params)
          action.end(my_files)
          sys.runs[key] = my_files
          continuation(my_files)
          dep_count[0] = -1
      
      def my_continuation_for(dep_class):
        def impl(single_dep_files):
          dep_files.setdefault(dep_class, []).extend(single_dep_files)
          dep_count[0] -= 1
          check_done()
        return impl
      
      for (dep_class, dep) in all_deps:
        dep.run(sys, params, my_continuation_for(dep_class))
      
      check_done()
  def doit(self, sysentry, dep_files, params):
    """
    The doit function is called for all subclasses of this.  The doit
    function doesn't necessarily perform the action, but is required to
    figure out exactly what actions are required.  For instance, for the
    build system, this generates a Makefile rule, which contains the exact
    commands and file names.
    
    The following are sent to an implementors' doit:
      - dep_files: the files "generated" by all rules it depends on.  The rules
      are grouped into the label of the rule (given when you called
      Rule.__init__) and then into the type of the file (like file
      extension).  The rules you are dependent on were the ones responsible
      for assigning the file extension.  Each file has a "simple" name and
      an "absolute" name.  The simple name should be used for generating
      result files, and the absolute name is the location of the source
      file.
      - params: the parameterization of the instantiation; that is, each
      file can be created with combinations of different parameters, and the
      parameterization is used to identify different "builds" of the same
      file.  in the case of the build system, this is different operating
      systems or compilation modes.  certain rules care about certain
      parameters, so when they generate destination files, they specify
      which parameters they care about.
      - the state (an encapsulation of params and input files)
          TODO: Remember why this is here, it doesn't seem to be used anywhere
      - systentry: a single entry in thie build system.  This is equivalent
      to a Makefile rule waiting for you to put actions into it.
      Unlike this Rule class, this object wants exact specific commands
      for a specific parameterization, like a Makefile rule with real
      filenames and commands.
    You must return:
      - a list of key,val pairs, where key is some indicator of file type. 
      Example is [("header", "foo.h"), ("source", "foo.cc")].
    """
    # TODO: Implement me in subclasses
    return util.map_values(FileCollection.to_pairs, dep_files)

class CustomRule(Rule):
  def __init__(self, dep_map, doit_fn):
    Rule.__init__(self, **dep_map)
    self.doit_fn = doit_fn
  def doit(self, sysentry, dep_files, params):
    return self.doit_fn(sysentry, dep_files, params)
