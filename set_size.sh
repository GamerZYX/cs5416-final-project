#!/bin/bash

# Define the directory where the database source files are located
DOC_DIR="./documents"

# --- Input Validation ---
if [ -z "$1" ]; then
    echo "Usage: $0 <suffix>"
    echo "Suffix must be 'l' or 's'."
    exit 1
fi
SUFFIX_LOWER=$(echo "$1" | tr '[:upper:]' '[:lower:]')
if [[ "$SUFFIX_LOWER" != "l" && "$SUFFIX_LOWER" != "s" ]]; then
    echo "Error: Invalid suffix '$1'. Suffix must be 'l' or 's'."
    exit 1
fi

# --- Calculate Absolute Paths ---
# Get the absolute path of the current directory
CURRENT_DIR=$(pwd)

# Define the source file names relative to the current directory
DB_REL_PATH="$DOC_DIR/documents_$SUFFIX_LOWER.db"
FAISS_REL_PATH="faiss_index$SUFFIX_LOWER.bin"

# Calculate the ABSOLUTE paths for the source files
DB_SRC_ABS="$CURRENT_DIR/$DB_REL_PATH"
FAISS_SRC_ABS="$CURRENT_DIR/$FAISS_REL_PATH"

# --- Define Link Destinations ---
# DB link is created INSIDE the documents folder
DB_LINK="$DOC_DIR/documents.db"
# FAISS link remains in the current directory
FAISS_LINK="faiss_index.bin"

# --- Create Symbolic Links ---

echo "Creating absolute symbolic links for suffix '$SUFFIX_LOWER'..."

# Link 1: documents.db (IN ./documents/)
if [ -f "$DB_REL_PATH" ]; then
    # Create the link (DB_LINK) pointing to the absolute path (DB_SRC_ABS)
    ln -sf "$DB_SRC_ABS" "$DB_LINK"
    echo "  -> Linked **$DB_LINK** to **$DB_SRC_ABS**"
else
    echo "  -> Error: Source file $DB_REL_PATH not found."
fi

# Link 2: faiss.bin (IN ./)
if [ -f "$FAISS_REL_PATH" ]; then
    # Create the link (FAISS_LINK) pointing to the absolute path (FAISS_SRC_ABS)
    ln -sf "$FAISS_SRC_ABS" "$FAISS_LINK"
    echo "  -> Linked **$FAISS_LINK** to **$FAISS_SRC_ABS**"
else
    echo "  -> Error: Source file $FAISS_REL_PATH not found."
fi

echo "Script finished."