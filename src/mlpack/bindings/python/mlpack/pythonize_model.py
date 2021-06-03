#!/usr/bin/env python
#
# Utilities for model parameters extraction.
#
# @file pythonize_model.py
# @author Mehul Kumar Nirala
#
# mlpack is free software; you may redistribute it and/or modify it under the
# terms of the 3-clause BSD license.  You should have received a copy of the
# 3-clause BSD license along with mlpack.  If not, see
# http://www.opensource.org/licenses/BSD-3-Clause for more information.


import numpy as np
from collections import defaultdict
from xml.etree import cElementTree as ET

def etree_to_dict(t, attrib = False):
  """
  Convert ElementTree to python dict.
  """
  d = {t.tag: {} if t.attrib else None}
  children = list(t)
  if children:
    dd = defaultdict(list)
    for dc in map(etree_to_dict, children):
      for k, v in dc.items():
        dd[k].append(v)
    d = {t.tag: {k:v[0] if len(v) == 1 else v for k, v in dd.items()}}
  if t.attrib and attrib:
    d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
  if t.text:
    text = t.text.strip()
    if children or t.attrib:
      if text:
        d[t.tag]['#text'] = text
    else:
      d[t.tag] = text
  return d

# Vector state mapping.
vec_state = {0: "matrix", 1: "column vector", 2: "row vector"}

def recursive_transform(json):
  """
  Recursiely traverse and convert array to numpy
  std::pair<> to python tuple().
  """
  # Check if the parameter is dict.
  if type(json).__name__ == 'dict':
    keys = list(json.keys())

    # Check if serialized object was a std::pair<>, std::unordered_map.
    if any(key in keys for key in ['first', 'second']):
      pair = (recursive_transform(json['first']), recursive_transform(json['second']))
      for key in ['first', 'second']:
        del json[key]
      return pair

    # Check if the field containns Armadillo serialization numbers.
    if any(key in keys for key in ['n_cols', 'n_rows', 'n_elem']):
      if 'item' in keys:
        json['item'] = np.array(map(float, json['item']))
      try:
        json['vec_state'] = vec_state[int(json['vec_state'])]
      except Exception as e:
        print(e, 'Vec state :%d, not known'%(int(json['vec_state'])))

    # Recursively call on the keys.
    for key in keys:
      json[key] = recursive_transform(json[key])

  # Check if the parameter is list.
  elif type(json).__name__ == 'list':
    item_list = []
    for x in json:
      item_list.append(recursive_transform(x))
    return item_list

  # The paramter is an item/value.
  else:
    try:
      # Convert to float if some numeric data is found.
      json = float(json)
    except Exception as e:
      pass
  return json

def transform(model):
  """
  Extract serialized information for model.
  """
  try:
    if type(model).__name__ != 'str':
      # Get the boost serializaled xml.
      model = model.__getparams__()
  except Exception as e:
    raise e

  # Get the xml tree.
  tree = ET.XML(model)

  # Convert the XML tree to python dict.
  model_dict = etree_to_dict(tree)
  model_dict = model_dict[list(model_dict.keys())[0]]
  model_dict = model_dict[list(model_dict.keys())[0]]
  return recursive_transform(model_dict)
