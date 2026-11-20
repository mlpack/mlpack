#!/usr/bin/env python
"""
preprocess_json_params.py: utility functions for json paramter preprocessing
                           (see set_cpp_param() and get_cpp_param() methods
                            in print_class_defn.hpp)

The "process_params_out" and "process_params_in" utilities are used to handle
interconversion between the output json from cereal and python dictionary.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
import numpy as np
import json
import pprint
from copy import deepcopy
from collections import OrderedDict

def process_params_out(model, params, return_str=False):
  '''
  This method processes the parameters obtained from the model.

  params:
  1) model - the model to process params.
  2) params - json parameters of the model (which we get through cereal).
  3) return_str (bool) - if True then a pretty string version of the 
                         params is returned.
  '''
  # for pretty printing.
  pp = pprint.PrettyPrinter()

  # value_resolver defined later.
  params_dic = json.loads(params, object_pairs_hook=value_resolver)

  # remove "cereal_class_version".
  # this stores the cereal_class_version value for all deleted pairs.
  cereal_class_version = []
  ref_path = []
  # 'full_paths' will store the complete path in the dictionary for all
  # occurrences. This will be used during the reversed process, to insert
  # 'cereal_class_version' at the correct places to avoid any errors.
  full_paths = []

  scrub(params_dic, "cereal_class_version", cereal_class_version, full_paths,
      ref_path)

  # storing 'cereal_class_version' occurrences paths and values.
  model.scrubbed_params["cereal_class_version"] = {
      "values": cereal_class_version,
      "full_paths": full_paths,
  }

  # convert armadillo dictionary to numpy array.
  arma_to_np(params_dic)

  if return_str:
    return params_dic, pp.pformat(params_dic)
  else:
    return params_dic

def process_params_in(model, params_dic):
  """
  This function takes in a model and the parameters dictionary,
  and returns a string that can be ingested back into the model.
  """
  # deepcopy to prevent changes to the user dictionary.
  params_dic_copy = deepcopy(params_dic)

  # convert numpy to armadillo.
  np_to_arma(params_dic_copy)

  # inserting scrubbed parameters back into dictionary.
  for param_name, details in model.scrubbed_params.items():
    for val, path in zip(details["values"], details["full_paths"]):
      insert_in_dic(params_dic_copy, path, param_name, val)

  # dumping to string. restore_value defined later.
  params_str = json.dumps(params_dic_copy, cls=restore_value)
  return params_str

def np_to_arma(obj):
  """
  This function replaces a numpy array to json representation
  of armadillo vector. This is reverse of "arma_to_np(obj)".
  """
  if isinstance(obj, OrderedDict):
    for key in obj.keys():
      """
      Checking if this is a numpy array.
      """
      if isinstance(obj[key], np.ndarray):
        # n_rows, n_cols have to be strings.
        n_rows, n_cols = obj[key].shape

        dic = OrderedDict()

        dic["n_rows"] = str(n_cols) # implicit transpose
        dic["n_cols"] = str(n_rows) # implicit transpose

        if n_cols != 1 and n_rows != 1:
          dic["vec_state"] = "0"
        elif n_rows == 1:
          dic["vec_state"] = "1"
        elif n_cols == 1:
          dic["vec_state"] = "2"

        elems = obj[key].flatten()
        dic["elem"] = list(elems)
        obj[key] = dic
      else:
        np_to_arma(obj[key])
  elif isinstance(obj, list):
    for i in range(len(obj)):
      np_to_arma(obj[i])
  else:
    # we cannot recurse further if we do not have a
    # dictionary or list object, so just pass.
    pass

def arma_to_np(obj):
  """
  This function replaces the JSON representation of armadillo vector to
  numpy array in the given dictionary.
  """
  if isinstance(obj, OrderedDict):
    for key in obj.keys():
      if isinstance(obj[key], OrderedDict):
        # if "vec_state" is present in dictionary, then it must be an Armadillo
        # vector.
        if "vec_state" in obj[key].keys():
          n_rows = int(obj[key]["n_rows"])
          n_cols = int(obj[key]["n_cols"])

          # Perform an implicit transpose, if there are any elements.
          if n_rows > 0 and n_cols > 0:
            obj[key] = np.array(obj[key]["elem"]).reshape(n_cols,
                n_rows).astype(type(obj[key]["elem"][0]))
          else:
            obj[key] = np.zeros((n_rows, n_cols))
        else:
          arma_to_np(obj[key])
      else:
        arma_to_np(obj[key])
  elif isinstance(obj, list):
    for i in range(len(obj)):
      arma_to_np(obj[i])
  else:
    # we cannot recurse further if we do not have a
    # dictionary or list object, so just pass.
    pass

def scrub(obj, bad_key, values, full_paths, ref_path):
  """
  This function removes a certain key-value pair from the
  given dictionary.
  params:
  1) obj (dict) - dictionary to traverse.
  2) bad_key (str) - key to remove.
  3) values (list) - list of values of all occurrences of bad_key
                    (this will be used to insert bad_key back into dictionary).
  4) full_paths (list) - this is a list that contains full path to all
                         occurrences of bad_key (used to insert bad_key back 
                         into dictionary).
  5) ref_path (list) - this for keeping track of the current path in the
                       dictionary.
  """
  if isinstance(obj, OrderedDict):
    for key in list(obj.keys()):
      ref_path.append(key)
      if key == bad_key:
        ref_path.pop()
        ref_path_copy = deepcopy(ref_path)
        full_paths.append(ref_path_copy)
        values.append(obj[key])
        del obj[key]
      else:
        scrub(obj[key], bad_key, values, full_paths, ref_path)
    if ref_path != []:
      ref_path.pop()  
  elif isinstance(obj, list):
    for i in range(len(obj)):
      ref_path.append(f"listidx_{i}")
      scrub(obj[i], bad_key, values, full_paths, ref_path)
    if ref_path != []:
      ref_path.pop()  
  else:
    ref_path.pop()
    pass

def value_resolver(pairs):
  '''
  This function converts multiple "elem" occurences to a list when
  used with json.loads().
  Eg:
  str({
    vec_state: 1,
    n_rows: 2,
    n_cols: 1,
    elem: 1,
    elem: 2
  })

  will be converted to

  dict({
    vec_state: 1,
    n_rows: 2,
    n_cols: 1,
    elem: [1,2]
  })
  This is done to handle same keys in the json while converting to python
  dictionary.
  '''
  has_elem = False
  for key,val in pairs:
    if key == "elem":
      has_elem = True
      break
  if has_elem:
    val_list = [val for (key,val) in pairs if key == "elem"]
    pairs = [(key,val) for (key,val) in pairs if key != "elem"]
    pairs.append(("elem", val_list))
  return OrderedDict(pairs)

class restore_value(json.JSONEncoder):
  '''
  This is a custom encoder that converts a dictionary to
  correct json format for ingesting in cereal.
  Eg:
  dict({
    vec_state: 1,
    n_rows: 2,
    n_cols: 1,
    elem: [1,2]
  })

  will be converted into

  str({
    vec_state: 1,
    n_rows: 2,
    n_cols: 1,
    elem: 1,
    elem: 2
  })
  while encoding.
  This is used to create a json that can be ingested to cereal.
  '''
  def encode(self, o):
    if isinstance(o, dict):
      if "elem" in o.keys():
        to_return = '{%s' % ', '.join(
            ': '.join((json.encoder.py_encode_basestring(k), self.encode(v)))\
                for k, v in o.items() if k != "elem")
        for val in o["elem"]:
          to_return += ', ' + json.encoder.py_encode_basestring("elem") +\
              f': {val}'
        to_return += "}"
        return to_return
      else:
        to_return = '{%s}' % ', '.join(
            ': '.join((json.encoder.py_encode_basestring(k), self.encode(v)))\
                for k, v in o.items())
        return to_return
    if isinstance(o, list):
      to_return = '[%s]' % ', '.join((self.encode(k) for k in o))
      return to_return
    return super().encode(o)

def insert_in_dic(dic, path, key, val):
  '''
  This function inserts a particluar key-value pair in a dictionray
  after following a particular path.
  '''
  temp = dic[path[0]]
  for idx in range(1, len(path)):
    if "listidx_" in path[idx]:
      temp = temp[int(path[idx].replace("listidx_", ""))]
    else:
      temp = temp[path[idx]]
  temp[key] = val
  # moving key-value pair to the start.
  temp.move_to_end(key, last=False)
