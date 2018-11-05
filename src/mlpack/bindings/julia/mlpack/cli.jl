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
      Cstring), paramName, paramValue);
end

function CLISetParamMat(paramName::String,
                        paramValue::Array{Float64, 2},
                        pointsAsRows::Bool)
  ccall((:CLI_SetParamMat, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Ptr{Float64}, UInt64, UInt64, Bool), paramName, Base.pointer(paramValue),
      size(paramValue, 1), size(paramValue, 2), pointsAsRows);
end

function CLISetParamUMat(paramName::String,
                         paramValue::Array{UInt64, 2},
                         pointsAsRows::Bool)
  ccall((:CLI_SetParamUMat, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Ptr{UInt64}, UInt64, UInt64, Bool), paramName, Base.pointer(paramValue),
      size(paramValue, 1), size(paramValue, 2), pointsAsRows);
end

function CLISetParam(paramName::String,
                     vector::Vector{String})
  # For this we have to set the size of the vector then each string
  # sequentially.  I am not sure if this is fully necessary but I have some
  # reservations about Julia's support for passing arrays of strings correctly
  # as a const char**.
  ccall((:CLI_SetParamVectorStrLen, "./libmlpack_julia_util.so"), Nothing,
      (Cstring, UInt64), paramName, size(vector, 1));
  for i in 1:size(vector, 1)
    ccall((:CLI_SetParamVectorStrStr, "./libmlpack_julia_util.so"), Nothing,
        (Cstring, Cstring, UInt64), paramName, vector[i], i - 1);
  end
end

function CLISetParam(paramName::String,
                     vector::Vector{Int})
  CLISetParam(paramName, convert(Vector{Int64}, vector))
end

function CLISetParam(paramName::String,
                     vector::Vector{Int64})
  ccall((:CLI_SetParamVectorInt, "./libmlpack_julia_util.so"), Nothing,
      (Cstring, Ptr{Int64}, Int64), paramName, Base.pointer(vector),
      size(vector, 1));
end

function CLISetParam(paramName::String,
                     matWithInfo::Tuple{Array{Bool, 1}, Array{Float64, 2}},
                     pointsAsRows::Bool)
  ccall((:CLI_SetParamMatWithInfo, "./libmlpack_julia_util.so"), Nothing,
      (Cstring, Ptr{Bool}, Ptr{Float64}, Int64, Int64, Bool), paramName,
      Base.pointer(matWithInfo[1]), Base.pointer(matWithInfo[2]),
      size(matWithInfo[2], 1), size(matWithInfo[2], 2), pointsAsRows);
end

function CLISetParamRow(paramName::String,
                        paramValue::Array{Float64, 1})
  ccall((:CLI_SetParamRow, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Ptr{Float64}, UInt64), paramName, Base.pointer(paramValue),
      size(paramValue, 1));
end

function CLISetParamCol(paramName::String,
                        paramValue::Array{Float64, 1})
  ccall((:CLI_SetParamCol, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Ptr{Float64}, UInt64), paramName, Base.pointer(paramValue),
      size(paramValue, 1));
end

function CLISetParamURow(paramName::String,
                         paramValue::Array{UInt64, 1})
  ccall((:CLI_SetParamURow, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Ptr{UInt64}, UInt64), paramName, Base.pointer(paramValue),
      size(paramValue, 1));
end

function CLISetParamUCol(paramName::String,
                         paramValue::Array{UInt64, 1})
  ccall((:CLI_SetParamUCol, "./libmlpack_julia_util.so"), Nothing, (Cstring,
      Ptr{UInt64}, UInt64), paramName, Base.pointer(paramValue),
      size(paramValue, 1));
end

function CLIGetParamBool(paramName::String)
  return ccall((:CLI_GetParamBool, "./libmlpack_julia_util.so"), Bool,
      (Cstring,), paramName)
end

function CLIGetParamInt(paramName::String)
  return ccall((:CLI_GetParamInt, "./libmlpack_julia_util.so"), Int64,
      (Cstring,), paramName)
end

function CLIGetParamDouble(paramName::String)
  return ccall((:CLI_GetParamDouble, "./libmlpack_julia_util.so"), Float64,
      (Cstring,), paramName)
end

function CLIGetParamString(paramName::String)
  return ccall((:CLI_GetParamString, "./libmlpack_julia_util.so"), Cstring,
      (Cstring,), paramName)
end

function CLIGetParamVectorStr(paramName::String)
  local size::UInt64
  local ptr::Ptr{String}

  # Get the size of the vector, then each element.
  size = ccall((:CLI_GetParamVectorStrLen, "./libmlpack_julia_util.so"), UInt64,
      (Cstring,), paramName);
  out = Array{String, 1}()
  for i = 1:size
    s = ccall((:CLI_GetParamVectorStrStr, "./libmlpack_julia_util.so"), Cstring,
        (Cstring, UInt64), paramName, i - 1)
    push!(out, Base.unsafe_string(s))
  end

  return out
end

function CLIGetParamVectorInt(paramName::String)
  local size::UInt64
  local ptr::Ptr{Int64}

  # Get the size of the vector, then the pointer to it.  We will own the
  # pointer.
  size = ccall((:CLI_GetParamVectorIntLen, "./libmlpack_julia_util.so"), UInt64,
      (Cstring,), paramName);
  ptr = ccall((:CLI_GetParamVectorIntPtr, "./libmlpack_julia_util.so"),
      Ptr{Int64}, (Cstring,), paramName);

  return Base.unsafe_wrap(Array{Int64, 1}, ptr, (size), own=true)
end

function CLIGetParamMat(paramName::String, pointsAsRows::Bool)
  # Can we return different return types?  For now let's restrict to a matrix to
  # make it easy...
  local ptr::Ptr{Float64}
  local rows::UInt64, cols::UInt64;
  # I suppose it would be possible to do this all in one call, but this seems
  # easy enough.
  rows = ccall((:CLI_GetParamMatRows, "./libmlpack_julia_util.so"), UInt64,
      (Cstring,), paramName);
  cols = ccall((:CLI_GetParamMatCols, "./libmlpack_julia_util.so"), UInt64,
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

function CLIGetParamUMat(paramName::String, pointsAsRows::Bool)
  # Can we return different return types?  For now let's restrict to a matrix to
  # make it easy...
  local ptr::Ptr{UInt64}
  local rows::UInt64, cols::UInt64;
  # I suppose it would be possible to do this all in one call, but this seems
  # easy enough.
  rows = ccall((:CLI_GetParamUMatRows, "./libmlpack_julia_util.so"), UInt64,
      (Cstring,), paramName);
  cols = ccall((:CLI_GetParamUMatCols, "./libmlpack_julia_util.so"), UInt64,
      (Cstring,), paramName);
  ptr = ccall((:CLI_GetParamUMat, "./libmlpack_julia_util.so"), Ptr{UInt64},
      (Cstring,), paramName);

  if pointsAsRows
    # In this case we have to transpose, unfortunately.
    m = Base.unsafe_wrap(Array{UInt64, 2}, ptr, (rows, cols), own=true)
    return m';
  else
    # Here no transpose is necessary.
    return Base.unsafe_wrap(Array{UInt64, 2}, ptr, (rows, cols), own=true);
  end
end

function CLIGetParamCol(paramName::String)
  local ptr::Ptr{Float64};
  local rows::UInt64;

  rows = ccall((:CLI_GetParamColRows, "./libmlpack_julia_util.so"), UInt64,
      (Cstring,), paramName);
  ptr = ccall((:CLI_GetParamCol, "./libmlpack_julia_util.so"), Ptr{Float64},
      (Cstring,), paramName);

  return Base.unsafe_wrap(Array{Float64, 1}, ptr, rows, own=true);
end

function CLIGetParamRow(paramName::String)
  local ptr::Ptr{Float64};
  local cols::UInt64;

  cols = ccall((:CLI_GetParamRowCols, "./libmlpack_julia_util.so"), UInt64,
      (Cstring,), paramName);
  ptr = ccall((:CLI_GetParamRow, "./libmlpack_julia_util.so"), Ptr{Float64},
      (Cstring,), paramName);

  return Base.unsafe_wrap(Array{Float64, 1}, ptr, cols, own=true);
end

function CLIGetParamUCol(paramName::String)
  local ptr::Ptr{UInt64};
  local rows::UInt64;

  rows = ccall((:CLI_GetParamUColRows, "./libmlpack_julia_util.so"), UInt64,
      (Cstring,), paramName);
  ptr = ccall((:CLI_GetParamUCol, "./libmlpack_julia_util.so"), Ptr{UInt64},
      (Cstring,), paramName);

  return Base.unsafe_wrap(Array{UInt64, 1}, ptr, rows, own=true);
end

function CLIGetParamURow(paramName::String)
  local ptr::Ptr{UInt64};
  local cols::UInt64;

  cols = ccall((:CLI_GetParamURowCols, "./libmlpack_julia_util.so"), UInt64,
      (Cstring,), paramName);
  ptr = ccall((:CLI_GetParamURow, "./libmlpack_julia_util.so"), Ptr{UInt64},
      (Cstring,), paramName);

  return Base.unsafe_wrap(Array{UInt64, 1}, ptr, cols, own=true);
end

function CLIGetParamMatWithInfo(paramName::String, pointsAsRows::Bool)
  local ptrBool::Ptr{Bool};
  local ptrData::Ptr{Float64};
  local rows::UInt64;
  local cols::UInt64;

  rows = ccall((:CLI_GetParamMatWithInfoRows, "./libmlpack_julia_util.so"),
      UInt64, (Cstring,), paramName);
  cols = ccall((:CLI_GetParamMatWithInfoCols, "./libmlpack_julia_util.so"),
      UInt64, (Cstring,), paramName);
  ptrBool = ccall((:CLI_GetParamMatWithInfoBoolPtr,
      "./libmlpack_julia_util.so"), Ptr{Bool}, (Cstring,), paramName);
  ptrMem = ccall((:CLI_GetParamMatWithInfoPtr,
      "./libmlpack_julia_util.so"), Ptr{Float64}, (Cstring,), paramName);

  types = Base.unsafe_wrap(Array{Bool, 1}, ptrBool, (rows), own=true)
  if pointsAsRows
    # In this case we have to transpose, unfortunately.
    m = Base.unsafe_wrap(Array{Float64, 2}, ptr, (rows, cols), own=true)
    return (types, m');
  else
    # Here no transpose is necessary.
    return (types, Base.unsafe_wrap(Array{Float64, 2}, ptr, (rows, cols),
        own=true));
  end
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
