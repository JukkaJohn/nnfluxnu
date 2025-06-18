FILES=("fit_sibyll.py")

# Loop over files and perform replacement
for FILE in "${FILES[@]}"; do
  if [ -f "$FILE" ]; then
    echo "Updating $FILE..."
    # Replace all occurrences of 'final' with 'fine' in-place (no backup)
    sed -i '' 's/final/fine/g' "$FILE"
  else
    echo "File not found: $FILE"
  fi
done

echo "All replacements complete."

