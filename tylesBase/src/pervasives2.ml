include Pervasives

let (@) = ExtList.(@)
let identity x = x
let (<<-) f g x = f (g x)
let (->>) f g x = g (f x)
let (&) f x = f x
let flip f b a = f a b
let open_out_safe = open_out_gen [Open_wronly; Open_creat; Open_excl; Open_text] 0o666
let output_endline cout s = output_string cout s; output_string cout "\n"
let eps_float v = ldexp epsilon_float (snd (frexp v) - 1)

let try_finally f g x =
  match try `V(f x) with e -> `E e with
    | `V f_x -> g x; f_x
    | `E e -> (try g x with _ -> ()); raise e

let string_of_float v =
  let ans = string_of_float v in
    if ans.[String.length ans - 1] = '.' then ans ^ "0" else ans

let print_float = print_string <<- string_of_float

let float_of_stringi s =
  try float_of_int (int_of_string s)
  with Failure _ -> float_of_string s
