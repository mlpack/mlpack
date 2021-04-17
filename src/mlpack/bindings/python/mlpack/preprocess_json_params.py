from random import randint
import numpy as np
import json
import pprint
from copy import deepcopy

def process_params(model, return_str=False, pretty_print=False, remove_version=False):
  params = model.get_params()
  params_decoded = params.decode("utf-8").replace("true", "True")\
    .replace("false", "False")

  # this is to handle same key names of "elem".
  # same key values cannot exist in python dictionary,
  # so I am replacing "elem" with random numbers.
  str_to_find = '"elem":'
  res = [i for i in range(len(params_decoded)) if\
            params_decoded.startswith(str_to_find, i)]

  # this variable keeps track of what random numbers are generated,
  # to avoid same random numbers generated for two "elem" keys.
  gen_nums = []

  for i in range(len(res)):
    random_num = random_with_N_digits(4)

    # keep generating until unique number is not found.
    while(random_num in gen_nums):
      random_num = random_with_N_digits(4)

    params_decoded = params_decoded[:res[i]] + '"{}":'.format(random_num) +\
                  params_decoded[res[i]+len(str_to_find):]

  # now we can convert it to a python dictionary.
  params_dic = eval(params_decoded)

  # remove "cereal_class_version".
  if remove_version:
    scrub(params_dic, "cereal_class_version")

  # convert armadillo dictionary to numpy array
  arma_to_np(params_dic)

  pp = pprint.PrettyPrinter()

  if pretty_print:
    pp.pprint(params_dic)

  if return_str:
    return params_dic, pp.pformat(params_dic)
  else:
    return params_dic

def feed_params(model, params_dic):
  """
  This function takes in a model and the parameters dictionary,
  and sets the parameters of the model as the given parameters.
  """
  # deepcopy to prevent changes to the user dictionary.
  params_dic_copy = deepcopy(params_dic)

  # this list for keeping track of the random numbers generated to replace
  # '"elem":' string, because python dictionaries cannot hold same keys.
  rand_gen = []
  np_to_arma(params_dic_copy, rand_gen)

  # dumping to string.
  params_str = json.dumps(params_dic_copy)

  # replacing random numbers with '"elem":' to match JSON given by cereal.
  for rand_num in rand_gen:
    params_str = params_str.replace('"{}":'.format(rand_num), '"elem":')

  # setting parameters to the model.
  model.set_params(params_str.encode("utf-8"))

def np_to_arma(obj, rand_gen):
  """
  This function replaces a numpy array to json representation
  of armadillo vector. This is reverse of "arma_to_np(obj)".
  """
  if isinstance(obj, dict):
    for key in obj.keys():
      """
      Checking if this is a numpy array.
      """
      if isinstance(obj[key], np.ndarray):
        # n_rows, n_cols have to be strings.
        n_rows, n_cols = str(1),str(1)

        dic = dict()

        if len(obj[key].shape) == 1:
          n_rows = obj[key].shape[0]
          dic["vec_state"] = str(1)
        elif len(obj[key].shape) == 2:
          n_rows, n_cols = obj[key].shape
          dic["vec_state"] = str(2)
        else:
          raise RuntimeError("Invalid number of dimensions in array {}".format(len(onj[key].shape)))

        dic["n_rows"] = str(n_cols) # implicit transpose
        dic["n_cols"] = str(n_rows) # implicit transpose

        elems = obj[key].flatten().astype(float)

        # writing elements of vector with random generated keys,
        # these keys will be replaced by '"elem":' in "feed_params()" function.
        for elem in elems:
          random_key = random_with_N_digits(4)
          while(random_key in rand_gen):
            random_key = random_with_N_digits(4)
          rand_gen.append(random_key)
          dic[str(random_key)] = elem

        obj[key] = dic
      else:
        np_to_arma(obj[key], rand_gen)
  elif isinstance(obj, list):
    for i in range(len(obj)):
      np_to_arma(obj[i], rand_gen)
  else:
    pass

def arma_to_np(obj):
  """
  This function replaces the JSON representation of armadillo vector to
  numpy array in the given dictionary.
  """
  if isinstance(obj, dict):
    for key in obj.keys():
      if isinstance(obj[key], dict):
        # if "vec_state" is present in dictionary, then
        # it must be armadillo vector. 
        if "vec_state" in obj[key].keys():
          n_rows = int(obj[key]["n_rows"])
          n_cols = int(obj[key]["n_cols"])
          elem_keys = list(set(obj[key].keys()).difference(set(["n_rows", "n_cols", "vec_state"])))
          elems = []
          for elem in elem_keys:
            elems.append(obj[key][elem])

          if n_rows*n_cols != len(elems):
            raise RuntimeError("Shape {}x{} not valid with number of elements {}"
                .format(n_rows, n_cols, len(elems)))

          elems = np.array(elems).reshape(n_cols, n_rows).astype(float)
          obj[key] = elems
        else:
          arma_to_np(obj[key])
      else:
        arma_to_np(obj[key])
  elif isinstance(obj, list):
    for i in range(len(obj)):
      arma_to_np(obj[i])
  else:
    pass

def scrub(obj, bad_key):
  """
  This function removes a certain key-value pair from the
  given dictionary.
  """
  if isinstance(obj, dict):
    for key in list(obj.keys()):
      if key == bad_key:
        del obj[key]
      else:
        scrub(obj[key], bad_key)
  elif isinstance(obj, list):
    for i in range(len(obj)):
      if obj[i] == bad_key:
        del obj[i]
      else:
        scrub(obj[i], bad_key)
  else:
    pass

def random_with_N_digits(n):
  """
  Generates random N digit numbers.
  """
  range_start = 10**(n-1)
  range_end = (10**n)-1
  return randint(range_start, range_end)