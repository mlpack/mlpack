"""
FastExec client library very similar to the C library.

This provides command-line argument parsing in FASTexec style, and also provides
a global datastore for exporting metrics (and dumping them to a file if
desired).
"""

import datastore

import sys

"""The root datanode for the fx system."""
datanode = datastore.DataNode()

"""Extra command-line arguments that do not contain --."""
extra_args = []

def param(name, mustexist = True):
  v = datanode.get_val_path("/info/params/" + name.strip("/"))
  if mustexist and v == None:
    raise "Parameter [%s] not specified." % (name)
  return v

def param_exists(name):
  return param(name, False) != None

def param_default(name, val):
  if not param_exists(name):
    datanode.set_val_path("/info/params/" + name.strip("/"), val)

def param_str(name):
  # TODO: Validate?
  return param(name)

def param_int(name):
  # TODO: Validate?
  return int(param(name))

def param_float(name):
  return float(param(name))

def param_bool(name):
  str = param(name)
  if str == None or (len(str) > 0 and str[0] in "fFnN0"):
    return False
  else:
    return True

def dump():
  stream = sys.stdout
  fname = param("fx/output")
  if fname != None:
   stream = open(fname, "w")
  try:
    datanode.write_pathstyle(stream)
  finally:
    if fname != None:
      stream.close()

# TODO: Configurable

# TODO: FINISH THIS

def metric_set(instancename, name, val):
  if instancename == None:
    path = ["info", "metrics", name]
  else:
    path = ["instances", instancename, "metrics", name]
  datanode.set_val(path, val)

def subparam_set(instancename, name, val):
  path = ["instances", instancename, "params", name]
  datanode.set_val(path, val)

def __parse_args():
  """Parses command-line arguments.
  
  Note that the keys are in path-style and unescaped as so.
  However, the values are assumed not to be escaped.
  """
  global extra_args
  kvpairs = []
  for arg in sys.argv[1:]:
    if arg[0:2] == "--":
      if '=' in arg:
        i = arg.index("=")
        kvpairs.append((arg[2:i], arg[i+1:]))
      else:
        kvpairs.append((arg[2:], "1"))
    else:
      extra_args.append(arg)
  for (name, val) in kvpairs:
    param_default(name, val)

__parse_args()

if __name__ == "__main__":
  print datanode
  print extra_args
