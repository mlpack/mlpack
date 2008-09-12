let msg ?(pre="MSG") ?pos msg =
  match pos with
    | None -> pre ^ ": " ^ msg
    | Some p -> pre ^ "[" ^ Pos.to_string p ^ "] " ^ msg

let err = msg ~pre:"ERROR"
let warn = msg ~pre:"WARNING"
let bug = msg ~pre:"BUG"

let print_msg ?pre ?pos m =
  print_endline(
    match pre,pos with
      | (None, None) -> msg m
      | (Some pre, None) -> msg ~pre m
      | (None, Some pos) -> msg ~pos m
      | (Some pre, Some pos) -> msg ~pre ~pos m
  )

let print_err = print_msg ~pre:"ERROR"
let print_warn = print_msg ~pre:"WARNING"
let print_bug = print_msg ~pre:"BUG"

let max_array_length_error = "Out of memory, possibly because trying to construct array of size greater than " ^ (string_of_int Sys.max_array_length)
