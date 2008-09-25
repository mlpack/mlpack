(* printing common datatypes *)
let show_unit x = "()"
let show_bool x = if x then "true" else "false"
let show_pair a b (x,y) = "(" ^ a x ^ "," ^ b y ^ ")"
let show_option a x = match x with Some y -> "(Some " ^ a y ^")" | None -> "None" 
let show_int x = string_of_int x
let show_float x = string_of_float x
let show_list a xs = "[" ^ String.concat "," (List.map a xs) ^ "]" 

