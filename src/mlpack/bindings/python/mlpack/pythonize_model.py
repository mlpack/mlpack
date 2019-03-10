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

def xml_to_dict(s):
    tree = ET.XML(s)
    d = etree_to_dict(tree)
    d = d[d.keys()[0]]
    d = d[d.keys()[0]]
    return d

def transform(model):
    result = {}
    try:
        if type(model).__name__ != 'str':
            model = model.__getparams__()
    except Exception as e:
        raise e
    
    model_dict = xml_to_dict(model)

    # network parameters
    try:
        result["parameters"] = model_dict["parameters"]["item"]
    except Exception as e:
        try:
            result["parameters"] = model_dict["parameters"]
        except Exception as e:
            raise e
    result['parameters'] = np.array(map(float,result['parameters']))

    # looking for other parameters
    for param in model_dict:
        # remove attributes and 'parameters'
        if ('parameters' == param):
            continue
        result[param] = model_dict[param]

    return result