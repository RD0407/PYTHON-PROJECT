# File Hash Comparison Tool

This simple Python script compares two files (for example, two PDFs) by computing their SHA-1 hash values.
If the files have identical content, their hashes will match. Otherwise, the hashes will be different—even if the files are only slightly different!
Features

    Checks if two files are identical (bit-for-bit) or not.

    Displays the file sizes and the SHA-1 hash for each file.

    Tells you clearly whether the files are identical or not.

## Usage

1. Prepare Your Files

    Put your files in the same directory as the script (for example, pdf1.pdf and pdf2.pdf).

2. Run the Script

    You can run the script in Jupyter Notebook, or as a Python script.

## How does it work?

    The script reads each file in small pieces (chunks) and updates the hash calculation for each chunk.

    At the end, the script prints out the SHA-1 hash for each file.

    If the hashes are exactly the same, your files are identical.

    If the hashes are different, the files are not identical (even if their sizes match).

## Why use it?

    To quickly check if files are really the same, even if they have different filenames.

    To detect if files have been modified, even by a single byte.

## Notes

    PDF files can sometimes look the same but be different at the binary level (because of metadata or timestamps inside the PDF).

    For comparing actual text content, you might need a PDF text extractor instead!
