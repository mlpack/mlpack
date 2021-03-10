def process_params(model, return_dic=False):
  params = model.params()
  params_decoded = params.decode("utf-8").replace("true", "True")\
    .replace("false", "False")
  if return_dic:
    dic = eval(params_decoded)
    return params_decoded, dic
  else:
    return params_decoded
