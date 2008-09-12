open TylesBase

type t = {file:string option; line:int option; col:int option}

exception Bad of string
let raise_bad msg = raise (Bad msg)

exception Undefined

let assert_well_formed t =
  if Option.is_some t.col && not (Option.is_some t.line) then raise_bad "cannot set column number without line number"
    
let f s = {file=Some s; line=None; col=None}
let l k = {file=None; line=Some k; col=None}
let fl s k = {file=Some s; line=Some k; col=None}
let lc k1 k2 = {file=None; line=Some k1; col=Some k2}
let flc s k1 k2 = {file=Some s; line=Some k1; col=Some k2}
let unknown = {file=None; line=None; col=None}

let file_exn t = match t.file with Some s -> s | None -> raise Undefined
let line_exn t = match t.line with Some s -> s | None -> raise Undefined
let col_exn t = match t.col with Some s -> s | None -> raise Undefined

let set_file t s = let ans = {t with file = Some s} in assert_well_formed ans; ans
let set_line t k = let ans = {t with line = Some k} in assert_well_formed ans; ans
let set_col t k = let ans = {t with col = Some k} in assert_well_formed ans; ans

let incrl t k =
  match t.line with
      None -> raise Undefined
    | Some l -> {t with line = Some (l+k)}

let to_string t =
  if Option.is_none t.file && Option.is_none t.line && Option.is_none t.col then
    "unknown_position"
  else
    let f =
      match t.file with 
          None -> "" 
        | Some s -> (match t.line with None -> s | Some _ -> s ^ ":")
    in
      
    let l =
      match t.line with
          None -> "" 
        | Some k -> (match t.col with None -> string_of_int k | Some _ -> string_of_int k ^ ".")
    in
      
    let c =
      match t.col with
          None -> ""               
        | Some k -> string_of_int k
    in
      f ^ l ^ c
