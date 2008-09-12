include DynArray

let pad_set darr idx v default =
  let pad_size = idx - DynArray.length darr + 1 in
    if pad_size <= 0 then
      DynArray.set darr idx v
    else
      (DynArray.append (DynArray.init pad_size (fun _ -> default)) darr;
       DynArray.set darr idx v)

