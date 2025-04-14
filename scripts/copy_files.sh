dest=${!#}

# Loop through all parameters (except the last one) and copy each file to the destination
for origin in "$@"; do
  # Skip the last parameter (which is the destination)
  if [ "$origin" != "$dest" ]; then
    cp "$origin" "$dest"
  fi
done