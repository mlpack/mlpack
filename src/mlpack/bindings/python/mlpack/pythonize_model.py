#!/usr/bin/env python
"""
pythonize_model.py: utilities for mdel parameters extraction

This file defines a transform function which takes in the model and returns a dict of parameters

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
import ast
import numpy as np

# We need a unicode type, but on python3 we don't have it.
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = str


# parse json to get model parameters
def transform(model):
    result = {}
    try:
        if type(model).__name__ != 'str':
            model = model.__getparams__()
    except Exception as e:
        raise e
    json = ast.literal_eval(model)
    model_json = json['boost_serialization']['t']

    # network parameters
    try:
        result["parameters"] = model_json["parameters"]["item"]
    except Exception as e:
        try:
            result["parameters"] = model_json["parameters"]
        except Exception as e:
            raise e
    result['parameters'] = np.array(map(float,result['parameters']))

    # looking for other parameters
    for param in model_json:
        # remove attributes and 'parameters'
        if '@' in param or 'parameters' == param: 
            continue
        result[param] = model_json[param]

    return result

