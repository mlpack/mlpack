module Stream = Stream2
open Pervasives2
open Printf

exception Error of (Pos.t * string)
let raise_error p m = raise (Error(p,m))

let copy_file ?first ?last in_file out_file =
  if in_file = out_file then
    failwith "in_file and out_file must be different"
  else
    let cin = open_in in_file in
    let sin = Stream.lines_of_channel cin in
    let cout = open_out_safe out_file in
      
    (* skip past initial first-1 lines *)
    let first = max 1 (match first with None -> 1 | Some x -> x) in
    let pred k _ = k < (first - 1) in
    let _ = Stream.skip_whilei pred sin in
      
    let pred k _ = match last with None -> true | Some x -> k < x in
    let sin = Stream.keep_whilei pred sin in
    let f s = output_string cout (s ^ "\n") in
    Stream.iter f sin;
    close_in cin;
    close_out cout

let fold_stream' ?(file="") ?(strict=true) f init cstr =
  let lines = Stream.lines_of_chars cstr in
  let f accum s =
    try f accum s
    with Failure msg ->
      let n = Stream.count lines in
      let pos = if file = "" then Pos.l n else Pos.fl file n in
      if strict then
        raise_error pos msg 
      else
        (print_string (Msg.err ~pos msg); 
         print_char '\n';
         accum)
  in
  Stream.fold f init lines
    
let fold_stream ?(strict=true) f init cstr =
  fold_stream' ~strict f init cstr
    
let fold_channel' ?(file="") ?(strict=true) f init cin =
  try_finally (fold_stream' ~file ~strict f init) ignore (Stream.of_channel cin)
    
let fold_channel ?(strict=true) f init cin =
  try_finally (fold_stream ~strict f init) ignore (Stream.of_channel cin)
    
let fold_string ?(strict=true) f init s = 
  try_finally (fold_stream ~strict f init) ignore (Stream.of_string s) 
    
let fold_file ?(strict=true) f init file =
  try try_finally (fold_channel' ~file ~strict f init) close_in (open_in file)
  with Error (p,m) -> raise_error (Pos.set_file p file) m

      
let of_stream f (cstr : char Stream.t) =
  let lines = Stream.lines_of_chars cstr in
  let g l s =  match f s with None -> l | Some v -> v::l in
  try List.rev (Stream.fold g [] lines)
  with Failure m -> raise_error (Pos.l (Stream.count lines)) m

let of_channel f cin =
  try_finally (of_stream f) ignore (Stream.of_channel cin)
    
let of_string f s = 
  try_finally (of_stream f) ignore (Stream.of_string s) 
    
let of_file f file =
  try try_finally (of_channel f) close_in (open_in file)
  with Error (p,m) -> raise_error (Pos.set_file p file) m
    

let to_channel f cout l =
  let g a = output_string cout (f a); output_char cout '\n' in
    List.iter g l

let to_file f file l =
  try_finally (fun cout -> to_channel f cout l) close_out (open_out_safe file)
    
let to_string f l =
  let ans = Buffer.create (List.length l * 100) in
  let rec loop = function
    | [] -> ()
    | a::l ->
        Buffer.add_string ans (f a);
        Buffer.add_char ans '\n';
        loop l
  in loop l; Buffer.contents ans

let to_stream f l = Stream.of_string (to_string f l)
