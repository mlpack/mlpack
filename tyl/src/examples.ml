open Tyl.Mp.Edsl
open Tyl.Mp.Show
open Printf
let map = List.map
let compile = Tyl.Mp.Compiler.compile

let x = name "x"
let w = name "w"

(* disjunction examples *)

let example0 = 
  minimize (x + w) 
    where [(x,continuous 10.0 100.0);(w,continuous 2.0 50.0)]
    subject_to (disj [x <= w ; x >= w + litR 4.0])

let xA = name "xA"
let xB = name "xB"

let example1 = 
  minimize (xA + xB) 
    where [(xA,continuous 0.0 10.0) ; (xB,continuous 0.0 10.0)]
    subject_to (disj [ conj [xA >= litR 1.0; xB >= litR 1.0; xA + xB <= litR 5.0] 
                     ; conj [litR 5.0 <= xA; xA <= litR 8.0; litR 4.0 <= xB; xB <= litR 7.0]  ])

(* the diet problem *) 

let buy   = map name ["x1";"x2";"x3";"x4"]
let costs = map litR [1.0;3.0;5.0;7.0]
let minA  = litR 2.0
let minB  = litR 4.0
let minC  = litR 6.0
let nutrA = map litR [10.0;20.0;30.0;40.0]
let nutrB = map litR [50.0;60.0;70.0;80.0]
let nutrC = map litR [90.0;100.0;110.0;120.0]

let diet = 
  minimize (sum (costs <*> buy))
    where (map (fun x -> (x,real)) buy)
    subject_to (conj
      [ minA <= sum (nutrA <*> buy)
      ; minB <= sum (nutrB <*> buy)
      ; minC <= sum (nutrC <*> buy)
      ])

(* boolean examples *)

let (b1,b2) = (name "b1",name "b2")

let dlf_example = 
  minimize (x + w)
    where [(x,real);(w,real)]
    subject_to (exists (b1,bool) 
                  (exists (b2,bool) 
                     (isTrue (b1 || not b2))  ))

let conj_example = 
  minimize (x + w)
    where [(x,real);(w,real)]
    subject_to (exists (b1,bool) 
                  (exists (b2,bool) 
                     (isTrue (b1 && b2))  ))


let print' p = printf "++++++++++++++++\n\n%s\n\n================\n\n%s\n\n" (showp p) (showp (compile p))

;; print' example0 
;; print' example1 
(* ;; print' diet *)
(* ;; print' dlf_example *)
(* ;; print' conj_example *)


let testc = conj [ exists (x,real) (x+w <= x)
                 ; exists (xA,int) (x >= w) ]

;; printf "%s\n\n%s\n\n" (showc testc) (showc (Tyl.Mp.Cnf.toPNF testc))

