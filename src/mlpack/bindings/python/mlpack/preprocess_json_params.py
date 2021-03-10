def preprocess_params(params, return_dic=False):
  params_decoded = params.decode("utf-8").replace("true", "True")\
    .replace("false", "False")
  if return_dic:
    dic = eval(params_decoded)
    return params_decoded, dic
  else:
    return params_decoded
