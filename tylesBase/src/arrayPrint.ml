let a ?(cout=stdout) ?(sep="\n") to_string arr =
  let n = Array.length arr in
  let print = output_string cout in
  for i = 0 to n - 2 do
    print (to_string arr.(i));
    print sep;
  done;
  if n > 0 then (print (to_string arr.(n-1)); print "\n")
    
let aa ?(cout=stdout) ?(sep="\t") to_string arr =
  let a = a ~cout ~sep to_string in
  let n = Array.length arr in
  for i = 0 to n-1 do
    a arr.(i);
  done
  
let i ?(cout=stdout) ?(sep="\n") = a ~cout ~sep string_of_int
let f ?(cout=stdout) ?(sep="\n") = a ~cout ~sep string_of_float
let s ?(cout=stdout) ?(sep="\n") = a ~cout ~sep (fun x -> x)

let ii ?(cout=stdout) ?(sep="\t") = aa ~cout ~sep string_of_int
let ff ?(cout=stdout) ?(sep="\t") = aa ~cout ~sep string_of_float
let ss ?(cout=stdout) ?(sep="\t") = aa ~cout ~sep (fun x -> x)
