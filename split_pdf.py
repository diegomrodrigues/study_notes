import os
from PyPDF2 import PdfReader, PdfWriter

def split_pdf(input_pdf, root_dir, config):
    # Create a PDF reader object
    reader = PdfReader(input_pdf)
    
    for folder_name, page_range in config.items():
        # Create folder if it doesn't exist
        if not os.path.exists(f"{root_dir}/{folder_name}"):
            os.makedirs(f"{root_dir}/{folder_name}")
        
        # Create a PDF writer object
        writer = PdfWriter()
        
        # Add pages to the writer
        for page_num in range(page_range[0] - 1, page_range[1]):
            writer.add_page(reader.pages[page_num])
        
        # Create the output filename
        output_filename = f"{folder_name}_{page_range[0]}-{page_range[1]}.pdf"
        output_path = os.path.join(root_dir, folder_name, output_filename)
        
        # Write the output PDF
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
        
        print(f"Created: {output_path}")

# Example usage
input_pdf = "math-deep.pdf"  # Replace with your input PDF file name
config = {
    #"Vector Spaces": (49, 104),
    #"Matrices": (113, 134),
    "Hadamard Matrices": (141, 162),
    "Direct Sums": (167, 197),
    "Determinants": (205, 237),
    "Gaussian Elimination": (243, 312),
    "Vector Norm and Matrix Norms": (323, 367),
    "Solving Linear Systems": (373, 395),
    "Dual Space": (399, 433),
    "Euclidean Spaces": (437, 479),
    "QR Decomposition": (491, 506),
    "Hermitian Spaces": (513, 548),
    "Eigenvectors & Eigenvalues": (553, 574),
    "Unit Quaternions and Rotations": (585, 605),
    "Spectral Theorems": (609, 638),
    "Computing Eigenvalues": (645, 673),
    "SVD & Polar Form": (731, 748),
    "Applications": (753, 783)
}

split_pdf(input_pdf, "Linear Algebra", config)