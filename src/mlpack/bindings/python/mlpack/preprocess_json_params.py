from random import randint
import numpy as np
import pprint

def process_params(model, return_str=False, pretty_print=False):
  params = model.params()
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

def arma_to_np(obj):
  """
  This function replaces the armadillo dictionary vector to
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