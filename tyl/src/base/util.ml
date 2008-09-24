let spf = Printf.sprintf
let (%) f g x = f (g x)
let ($) f x = f x
let id x = x
let const a b = a
let flip f x y = f y x
let any = List.exists
let all = List.for_all
let map = List.map
let unzip = List.split
let zip = List.combine
let foldl = List.fold_left
let foldl1 f = function (x::xs) -> foldl f x xs | _ -> assert false

let rec map3 f xs ys zs = match xs,ys,zs with
  | x::xs',y::ys',z::zs' -> f x y z :: map3 f xs' ys' zs'
  | _ -> []

let rec transpose xss = match xss with 
  | [] -> [] 
  | []::xss' -> transpose xss'      
  | (x::xs) :: xss' -> (x :: map List.hd xss') :: transpose (xs :: map List.tl xss')

let foreach xs f = List.concat (List.map f xs)
let rec enumFromTo a b = if a==b then [a] else a :: enumFromTo (a+1) b

(* constructors as functions *)
let some x = Some x
let cons x xs = x::xs
let pair x y = (x,y)
