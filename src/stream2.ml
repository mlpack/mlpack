include Stream

let lines_of_chars cstr =
  let f _ =
    match peek cstr with
      | None -> None
      | Some _ ->
          let ans = Buffer.create 100 in
          let rec loop () =
            try
              let c = next cstr in
              if c <> '\n' then (Buffer.add_char ans c; loop())
            with Failure -> ()
          in 
          loop();
          Some (Buffer.contents ans)
  in
  from f

let lines_of_channel cin =
  let f _ =
    try Some (input_line cin)
    with End_of_file -> None
  in Stream.from f
  
let is_empty s =
  match peek s with None -> true | Some _ -> false
    
let keep_whilei pred s =
  let f _ =
    match peek s with
      | None -> None
      | Some a ->
          if pred (count s) a
          then (junk s; Some a)
          else None
  in from f
    
let keep_while pred = keep_whilei (fun _ a -> pred a)
let truncate k = keep_whilei (fun j _ -> j < k)
  
let rec skip_whilei pred s =
  match peek s with
    | None -> ()
    | Some a ->
        if pred (count s) a
        then (junk s; skip_whilei pred s)
        else ()

let skip_while pred = skip_whilei (fun _ a -> pred a)

let one f s =
  match peek s with
      None -> raise Failure
    | Some a -> if f a then (junk s; a) else raise Failure
	  
let many f s =
  let rec started s =
    match peek s with
	None -> []
      | Some a -> if f a then (junk s; a::(started s)) else []
  in
    match peek s with
	None -> raise Failure
      | Some a -> if f a then started s else raise Failure
	    
let rec fold f accum s =
  match peek s with
      None -> accum
    | Some a -> (junk s; fold f (f accum a) s)
	          
let map f s =
  let f _ =
    try Some (f (next s))
    with Failure -> None
  in from f
    
let sub_stream mk stop sa =
  let should_junk sa =
    match peek sa with
      | Some a -> stop a
      | None -> false
  in
  
  let consume sa = while should_junk sa do junk sa done in
  
  let make_stream (_:int) =
    let f al a = a::al in
    if is_empty sa then None
    else
      let sa = keep_while (fun a -> not (stop a)) sa in
      let ans = List.rev (fold f [] sa) in
      (consume sa; Some (mk ans))
  in
  from make_stream
    
let to_array t =
  let ans = DynArray.create () in
  let _ = iter (DynArray.add ans) t in
    DynArray.to_array ans
      
let to_list t =
  List.rev (fold (fun l b -> b::l) [] t)
