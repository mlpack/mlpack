let compose cmp1 cmp2 a b =
  let c1 = cmp1 a b in
  let c2 = cmp2 a b in
  let num_true = List.fold_left (fun cnt b -> if b then cnt+1 else cnt) 0 in
  let bl = [c1 = Some (-1); c1 = Some 1; c2 = Some (-1); c2 = Some 1] in
    assert((num_true bl <= 1) || (c1 = Some 0 && c2 = Some 0));
    match c1 with
      | Some c1 -> c1
      | None ->
          match c2 with
            | Some c2 -> c2
            | None -> invalid_arg "neither partial order given relates given values"
                
let reverse cmp a b = -(cmp a b)
let reversep cmp a b = Option.map (~-) (cmp a b)

let totalify cmp =
  fun a b ->
    match cmp a b with
      | Some c -> c
      | None -> failwith "order relation not defined for given elements"
