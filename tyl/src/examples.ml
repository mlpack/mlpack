open Tyl.Mp.Edsl
open Tyl.Mp.Show
open Printf

let x = name "x"
let w = name "w"

let example0 = 
  minimize (x + w + litR 2.0 * x)
    where [(x,real);(w,real)]
    subject_to (x <= w)

let example1 = 
  minimize (x + w) 
    where [(x,real);(w,real)]
    subject_to ((x <= w) |/ (x >= w + litR 4.0))

;;

printf "%s\n\n" (showp example0) ;;
printf "%s\n\n" (showp example1) ;;
