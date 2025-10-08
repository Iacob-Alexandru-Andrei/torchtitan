#!/bin/bash

# This script recursively generates AGENTS.md files by running embargo on all
# non-underscore-prefixed folders, splits their lines, and stages the results.
# This consolidated process ensures a predictable two-step commit when changes occur.

# --- Configuration ---
PROJECT_PATH=$(git rev-parse --show-toplevel)
# Define the path to the line splitter script
LINE_SPLITTER_SCRIPT="$PROJECT_PATH/precommit-hooks/split_long_lines.py"

# --- Functions ---

# Process a single directory
process_directory() {
    local dir_path="$1"
    local relative_path="${dir_path#$PROJECT_PATH/}"

    echo "========================================="
    echo "Processing directory: $relative_path"
    echo "========================================="

    # Define paths - embargo always creates EMBARGO.md in the directory where it's run
    local embargo_file="$PROJECT_PATH/EMBARGO.md"
    local agents_file="$dir_path/AGENTS.md"

    echo "Step 1: Generating project summary (EMBARGO.md)..."
    # Run embargo on this specific directory (creates EMBARGO.md in PROJECT_PATH)
    (cd "$PROJECT_PATH" && embargo --format llm-optimized --input "$relative_path")

    # Check if the EMBARGO.md file was created successfully in project root
    if [ ! -f "$embargo_file" ]; then
        echo "Warning: embargo command failed to generate EMBARGO.md for $relative_path" >&2
        echo "Skipping this directory..."
        return 1
    fi

    echo "Step 2: Moving EMBARGO.md to target directory and renaming to AGENTS.md..."
    mv "$embargo_file" "$agents_file"

    echo "Step 3: Splitting long lines in AGENTS.md..."
    # Call the Python script to process the newly created file
    python3 "$LINE_SPLITTER_SCRIPT" "$agents_file"

    echo "Step 4: Staging the generated AGENTS.md file..."
    git add "$agents_file"

    echo "✓ Completed processing: $relative_path"
    echo ""
}

# --- Script Execution ---

echo "Starting recursive AGENTS.md generation..."
echo ""

# Define the torchtitan source directory
TORCHTITAN_DIR="$PROJECT_PATH/torchtitan"

# Check if torchtitan directory exists
if [ ! -d "$TORCHTITAN_DIR" ]; then
    echo "Error: torchtitan directory not found at $TORCHTITAN_DIR" >&2
    exit 1
fi

# Step 1: Generate top-level AGENTS.md (for entire project including torchtitan/)
echo "========================================="
echo "Processing project root (top-level AGENTS.md)"
echo "========================================="
echo "Step 1: Generating project summary (EMBARGO.md)..."
(cd "$PROJECT_PATH" && embargo --format llm-optimized --input torchtitan)

if [ ! -f "$PROJECT_PATH/EMBARGO.md" ]; then
    echo "Error: Failed to generate top-level EMBARGO.md" >&2
    exit 1
fi

echo "Step 2: Renaming EMBARGO.md to AGENTS.md..."
mv "$PROJECT_PATH/EMBARGO.md" "$PROJECT_PATH/AGENTS.md"

echo "Step 3: Splitting long lines in top-level AGENTS.md..."
python3 "$LINE_SPLITTER_SCRIPT" "$PROJECT_PATH/AGENTS.md"

echo "Step 4: Copying top-level AGENTS.md to torchtitan/AGENTS.md..."
cp "$PROJECT_PATH/AGENTS.md" "$TORCHTITAN_DIR/AGENTS.md"

echo "Step 5: Staging both AGENTS.md files..."
git add "$PROJECT_PATH/AGENTS.md"
git add "$TORCHTITAN_DIR/AGENTS.md"

echo "✓ Completed processing: Project root and torchtitan/"
echo ""

# Step 2: Generate custom AGENTS.md for subdirectories within torchtitan/
# Find all directories inside torchtitan/ that don't start with underscore or dot
# Exclude common non-code directories and the torchtitan root itself
find "$TORCHTITAN_DIR" -type d \
    ! -path "$TORCHTITAN_DIR" \
    ! -path "$TORCHTITAN_DIR/__pycache__/*" \
    ! -path "$TORCHTITAN_DIR/*/.*" \
    ! -path "$TORCHTITAN_DIR/*/__pycache__/*" \
    ! -name "_*" \
    ! -name ".*" \
    ! -name "__pycache__" \
    -print0 | while IFS= read -r -d '' dir; do

    # Check if directory name (basename) starts with underscore or dot
    dir_basename=$(basename "$dir")
    if [[ "$dir_basename" == _* ]] || [[ "$dir_basename" == .* ]]; then
        continue
    fi

    # Check if directory contains Python files
    if find "$dir" -maxdepth 1 -name "*.py" -print -quit | grep -q .; then
        process_directory "$dir"
    fi
done

# Cleanup: Remove any stray EMBARGO.md files that might be left
echo "Step 6: Cleaning up any stray EMBARGO.md files..."
find "$PROJECT_PATH" -name "EMBARGO.md" -type f -delete
echo "✓ Cleanup complete"
echo ""

echo "========================================="
echo "Script finished. All AGENTS.md files have been staged."
echo "========================================="
# Exit with 0 for success. Pre-commit handles aborting the commit
# automatically when it detects staged file modifications.
exit 0
