open List

let (@@) f g x = f (g x)
let ($) f x = f x
let flip f x y = f y x
let any = exists
let all = for_all
let foldl = fold_left
let foldl1 f = function (x::xs) -> foldl f x xs | _ -> assert false

let rec map3 f xs ys zs = match xs,ys,zs with
  | x::xs',y::ys',z::zs' -> f x y z :: map3 f xs' ys' zs'
  | _ -> []

let rec transpose xss = match xss with 
  | [] -> [] 
  | []::xss' -> transpose xss'      
  | (x::xs) :: xss' -> (x :: map hd xss') :: transpose (xs :: map tl xss')
