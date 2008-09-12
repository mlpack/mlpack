include Char
  
(* check if ascii code of c lies between n1 and n2 *)
let in_code_range c n1 n2 =
  let c = code c
  in n1 <= c && c <= n2
    
let is_digit c = in_code_range c 48 57
let is_hex_digit c = in_code_range c 48 57 || in_code_range c 97 102 || in_code_range c 65 70
let is_oct_digit c = in_code_range c 48 55
let is_lower c = in_code_range c 97 122
let is_upper c = in_code_range c 65 90
let is_letter c = is_lower c || is_upper c
let is_alpha_num c = is_letter c || is_digit c
let is_ascii c = in_code_range c 0 127
  
let to_string c = String.make 1 c
  
let to_int c =
  if is_digit c then (code c) - (code '0')
  else raise (Invalid_argument ("cannot convert character " ^ (to_string c) ^ " to int"))
    
let from_int k =
  if k >= 0 && k <= 9 then chr (code '0' + k)
  else raise (Invalid_argument ("cannot convert int " ^ (string_of_int k) ^ " to char"))
    
let is_space = String.contains " \t\r\n"
