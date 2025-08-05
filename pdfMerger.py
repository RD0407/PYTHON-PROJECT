import os
from PyPDF2 import PdfMerger
pdf_files= ["pdf1.pdf","pdf3.pdf"]

merger =PdfMerger()

for pdf in pdf_files:
    merger.append(pdf)

merger.write("merged_pdf.pdf")

merger.close()

print("Merging pdfs complete! The final product is merged_pdf")
