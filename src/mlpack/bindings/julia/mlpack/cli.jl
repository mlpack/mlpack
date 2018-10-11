using Printf

function CLIRestoreSettings(programName::String)
  ccall((:CLI_RestoreSettings, "./libmlpack_julia_util.so"), Nothing, (Cstring,),
      programName);
end

function CLISetParam(paramName::String, paramValue::Int)
  ccall((:CLI_SetParamInt, "./libmlpack_julia_util.so"), Nothing, (Cstring, Int),
      paramName, paramValue);
end

function CLISetParam(paramName::String, paramValue::Float64)
  ccall((:CLI_SetParamDouble, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Float64), paramName, paramValue);
end

function CLISetParam(paramName::String, paramValue::Bool)
  ccall((:CLI_SetParamBool, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Bool), paramName, paramValue);
end

function CLISetParam(paramName::String, paramValue::String)
  ccall((:CLI_SetParamString, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      String), paramName, paramValue);
end

function CLISetParam(paramName::String,
                     paramValue::Array{Float64, 2},
                     pointsAsRows::Bool)
  ccall((:CLI_SetParamMat, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Ptr{Float64}, Int64, Int64, Bool), paramName, Base.pointer(paramValue),
      size(paramValue, 1), size(paramValue, 2), pointsAsRows);
end

function CLISetParam(paramName::String,
                     paramValue::Array{Int64, 2},
                     pointsAsRows::Bool)
  ccall((:CLI_SetParamUMat, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Ptr{Int64}, Int64, Int64, Bool), paramName, Base.pointer(paramValue),
      size(paramValue, 1), size(paramValue, 2), pointsAsRows);
end

function CLISetParamURow(paramName::String,
                        paramValue::Array{Int64, 1})
  ccall((:CLI_SetParamURow, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Ptr{Int64}, Int64), paramName, Base.pointer(paramValue),
      size(paramValue, 1));
end

function CLISetParamUCol(paramName::String,
                        paramValue::Array{Int64, 1})
  ccall((:CLI_SetParamUCol, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Ptr{Int64}, Int64), paramName, Base.pointer(paramValue),
      size(paramValue, 1));
end

function CLIGetParamMat(paramName::String, pointsAsRows::Bool)
  # Can we return different return types?  For now let's restrict to a matrix to
  # make it easy...
  local ptr::Ptr{Float64}
  local rows::Int64, cols::Int64;
  # I suppose it would be possible to do this all in one call, but this seems
  # easy enough.
  rows = ccall((:CLI_GetParamMatRows, "./libmlpack_julia_util.so"), Int64,
      (Cstring,), paramName);
  cols = ccall((:CLI_GetParamMatCols, "./libmlpack_julia_util.so"), Int64,
      (Cstring,), paramName);
  ptr = ccall((:CLI_GetParamMat, "./libmlpack_julia_util.so"), Ptr{Float64},
      (Cstring,), paramName);

  if pointsAsRows
    # In this case we have to transpose, unfortunately.
    m = Base.unsafe_wrap(Array{Float64, 2}, ptr, (rows, cols), own=true)
    return m';
  else
    # Here no transpose is necessary.
    return Base.unsafe_wrap(Array{Float64, 2}, ptr, (rows, cols), own=true);
  end
end

function CLIGetParamURow(paramName::String)
  local ptr::Ptr{Int64};
  local cols::Int64;

  cols = ccall((:CLI_GetParamURowCols, "./libmlpack_julia_util.so"), Int64,
      (Cstring,), paramName);
  ptr = ccall((:CLI_GetParamURow, "./libmlpack_julia_util.so"), Ptr{Int64},
      (Cstring,), paramName);

  return Base.unsafe_wrap(Array{Int64, 1}, ptr, cols, own=true);
end

function CLIEnableVerbose()
  ccall((:CLI_EnableVerbose, "./libmlpack_julia_util.so"), Nothing, ());
end

function CLIDisableVerbose()
  ccall((:CLI_DisableVerbose, "./libmlpack_julia_util.so"), Nothing, ());
end

function CLISetPassed(paramName::String)
  ccall((:CLI_SetPassed, "./libmlpack_julia_util.so"), Nothing, (Cstring,),
      paramName);
end
