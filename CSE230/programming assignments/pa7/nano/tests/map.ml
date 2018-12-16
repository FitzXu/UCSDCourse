let rec map = 
  fun f -> fun xs ->
    if (null xs) then [] else (f (hd xs)) :: (map f (tl xs))
in
  map