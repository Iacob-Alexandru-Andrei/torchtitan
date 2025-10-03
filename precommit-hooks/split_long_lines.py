#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

# Define the maximum number of bytes allowed per line.
# Shells often have a limit (e.g., ARG_MAX), and 4000 is a safe, conservative value.
MAX_BYTES = 4000
# The UTF-8 byte representation of the '→' character.
ARROW_BYTES = "→".encode("utf-8")


def split_line(line_bytes):
    """
    Splits a byte string into a list of smaller byte strings, each not
    exceeding MAX_BYTES. The split is preferably made right after the last
    '→' operator found within the MAX_BYTES limit of a given segment.

    Args:
        line_bytes (bytes): The line to be split.

    Returns:
        list[bytes]: A list of byte strings, each representing a segment
                     of the original line.
    """
    chunks = []
    remaining_line = line_bytes

    # Continue chunking as long as the remaining part is too long.
    while len(remaining_line) > MAX_BYTES:
        # Define the window to search for the arrow (the first MAX_BYTES).
        search_window = remaining_line[:MAX_BYTES]

        # Find the position of the last arrow in that search window.
        # rfind returns -1 if the substring is not found.
        split_pos = search_window.rfind(ARROW_BYTES)

        if split_pos != -1:
            # If an arrow is found, the ideal split point is right after it.
            cut_point = split_pos + len(ARROW_BYTES)
        else:
            # Fallback: If no arrow is in the first MAX_BYTES, split at the
            # limit to ensure the line is processed and we don't loop forever.
            cut_point = MAX_BYTES

        # Add the chunk to the list and update the remaining part of the line.
        chunks.append(remaining_line[:cut_point])
        remaining_line = remaining_line[cut_point:]

    # Add the final part of the line (which is now under the limit).
    if remaining_line:
        chunks.append(remaining_line)

    return chunks


def process_file(filename):
    """
    Reads a file and splits any lines that are longer than MAX_BYTES.
    The file is modified in place.

    Args:
        filename (str): The path to the file to process.

    Returns:
        int: 1 if the file was modified, 0 otherwise.
    """
    try:
        # Open the file in binary read mode to correctly measure byte length.
        with open(filename, "rb") as f:
            lines = f.readlines()
    except IOError as e:
        # Handle file reading errors.
        print(f"Error reading file {filename}: {e}", file=sys.stderr)
        return 1

    modified = False
    new_content = bytearray()

    # Iterate through each line of the file.
    for line in lines:
        # Check if the line (including the newline character) exceeds the limit.
        if len(line) > MAX_BYTES:
            modified = True
            # Remove the original newline character before splitting, if it exists.
            line_to_split = line.rstrip(b"\n\r")
            chunks = split_line(line_to_split)
            # Join the chunks with newline characters.
            new_content.extend(b"\n".join(chunks))
            # Add a final newline to match the original line ending.
            if line.endswith((b"\n", b"\r")):
                new_content.extend(b"\n")
        else:
            # If the line is within the limit, add it to the new content as is.
            new_content.extend(line)

    # If any modifications were made, write the new content back to the file.
    if modified:
        try:
            print(f"Splitting lines over {MAX_BYTES} bytes in {filename}")
            # Open the file in binary write mode to overwrite it.
            with open(filename, "wb") as f:
                f.write(new_content)
            # Return 1 to indicate that a change was made.
            return 1
        except IOError as e:
            # Handle file writing errors.
            print(f"Error writing to file {filename}: {e}", file=sys.stderr)
            return 1

    # Return 0 if no changes were made.
    return 0


def main():
    """
    Main function to parse arguments and process files.
    """
    # Set up argument parsing.
    parser = argparse.ArgumentParser(
        description=f"Split lines in files that are longer than {MAX_BYTES} bytes."
    )
    parser.add_argument("filenames", nargs="*", help="The files to check and process.")
    args = parser.parse_args()

    # Process each file and track if any have been changed.
    return_code = 0
    for filename in args.filenames:
        if process_file(filename):
            return_code = 1

    # Exit with a non-zero status code if any files were modified.
    # This is standard for pre-commit hooks to signal a failure.
    return return_code


if __name__ == "__main__":
    sys.exit(main())
