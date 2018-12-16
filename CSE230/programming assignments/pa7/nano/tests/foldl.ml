let foldl = fun f -> fun b -> fun l -> 
  let rec worker = fun acc -> fun xs ->
    if (null xs) then acc else worker (f acc (hd xs)) (tl xs)
  in
    worker b l
in
  foldl