open Printf

let time f a =
  let init = Sys.time() in
  let b = f a in
  let delt = Sys.time() -. init in
    printf "finished in %.2f seconds\n" delt; flush stdout;
    b
      
let sf msg f a =
  let _ = print_string (msg ^ "... "); flush stdout in
  let b = f a in
  let _ = print_endline "finished" in
    b

let timesf msg f a =
  let init = Sys.time() in
  let _ = print_string (msg ^ "... "); flush stdout in
  let b = f a in
  let delt = Sys.time() -. init in
    printf "finished in %.2f seconds\n" delt; flush stdout;
    b

let get_time f a =
  let init = Sys.time() in
  let b = f a in
  let delt = Sys.time() -. init in
    (b, delt)

let repeat n f =
  if n < 1 then failwith "cannot execute a function less than 1 time"
  else
    fun a ->
      for i = 1 to n-1 do ignore (f a) done;
      f a
