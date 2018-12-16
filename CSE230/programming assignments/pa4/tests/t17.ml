let sumList = fun xs -> 
  let worker = fun acc -> fun nxt -> acc + nxt in
  foldl worker 0 xs in
let m = [1;3;4] in
sumList m