include ExtArray.Array
  
exception Different_array_size
  
let zip a b =
  if length a <> length b
  then raise Different_array_size
  else init (length a) (fun i -> a.(i), b.(i))
    
let unzip ab = (map fst ab, map snd ab)

let of_idx_array idxa default =
  let max_idx = fold_left (fun max (idx,_) -> if max > idx then max else idx) (-1) idxa in
  let ans = make (max_idx + 1) default in
    for i = 0 to Array.length idxa - 1 do
      let (idx,v) = idxa.(i) in ans.(idx) <- v
    done;
    ans
  
let of_list2 ll = map of_list (of_list ll)
let to_list2 aa = to_list (map to_list aa)
  
let for_alli f a =
  let len = length a in
  let rec helper i =
    if i >= len then true
    else if f i (get a i) then helper (i+1)
    else false
  in
    helper 0
      
let existsi f a =
  let len = length a in
  let rec helper i =
    if i >= len then false
    else if f i (get a i) then true
    else helper (i+1)
  in
    helper 0
      
let find_helper f a =
  let len = length a in
  let rec helper i =
    if i >= len then raise Not_found
    else if f i (get a i) then (i, get a i)
    else helper (i+1)
  in
    helper 0
      
let find' f a = snd (find_helper f a)
let findi' f a = fst (find_helper f a)
  
let unique cmp a =
  let a = copy a in (* don't mess with input *)
  let _ = sort cmp a in
  let len = length a in
  let rec adjEq i =
    if len - i <= 1 then false
    else (cmp (get a i) (get a (i+1)) = 0) || (adjEq (i+1))
  in
    not (adjEq 0)
      
let is_rectangular d =
  if length d <= 1 then
    true
  else
    let allLengths = map length d in
    let firstLength = get allLengths 0 in
    let lengthEqualsFirstLength k = k = firstLength in
      for_all lengthEqualsFirstLength allLengths
