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
input_pdf = "Math Theory of Deep Learning.pdf"
config = {
    "Feedfoward Neural Networks": (15, 21),
    "Universal Approximation": (22, 35),
    "Splines": (36, 43),
    "ReLU Neural Networks": (44, 88),
    "High-dimensional Approximations": (89, 102),
    "Interpolation": (103, 110),
    "Training Neural Networks": (111, 139),
    "Wide Neural Networks": (140, 165),
    "Loss Landscape Analysis": (166, 175),
    "Neural Networks Space": (176, 188),
    "Generalization": (189, 206),
    "Overparameterization": (207, 218),
    "Adversarial Examples": (219, 233)
}
folder = "Deep Learning"

split_pdf(input_pdf, folder, config)