PDF Merger (Python)

A simple Python script to merge multiple PDF files into a single document.
Perfect for combining scanned documents, reports, or notes into one PDF.
Requirements

    Python 3.x

    PyPDF2 library

Installation

First, install PyPDF2 (recommended inside a virtual environment):

pip install PyPDF2

Caution:
If you get an error like
ModuleNotFoundError: No module named 'PyPDF2'
even after installing, you might need to use a virtual environment (venv):

python3 -m venv myenv
source myenv/bin/activate    # On Windows: myenv\Scripts\activate
pip install PyPDF2

If running in Jupyter, you may need to use Jupyter inside the same environment you installed PyPDF2.
Usage

    Place your PDF files in the same directory as the script.

    Edit the list of PDFs to merge:

pdf_files = ["pdf1.pdf", "pdf3.pdf"]

Add as many files as you want (they will merge in this order).

Run the script:

    python pdf_merger.py

    Result:
    The merged file will be saved as merged_pdf.pdf in the current directory.

Example Code

from PyPDF2 import PdfMerger

pdf_files = ["pdf1.pdf", "pdf3.pdf"]
merger = PdfMerger()

for pdf in pdf_files:
    merger.append(pdf)

merger.write("merged_pdf.pdf")
merger.close()

print("Merging pdfs complete! The final product is merged_pdf.pdf")

Troubleshooting

    Make sure your PDF files exist and are not open in another program.

    Use exact filenames, including .pdf extension.

    If PyPDF2 still won’t install, check your Python environment and see the PyPDF2 documentation.
