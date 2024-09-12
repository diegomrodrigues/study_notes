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
input_pdf = "eisenstein-nlp-notes.pdf"
config = {
    #"Vector Spaces": (49, 104),
    "Linear Text Classification": (31, 64),
    "Nonlinear Text Classification": (65, 86),
    "Linguistic Classification": (87, 112),
    "Unsupervised Learning": (113, 142),
    "Language Models": (143, 162),
    "Senquence Labeling": (163, 192),
    "Linguistic Sequence Labeling": (193, 208),
    "Formal Language Theory": (209, 242),
    "Context-free Parsing": (243, 274),
    "Dependency Parsing": (275, 299),
    "Logical Semantics": (303, 322),
    "Predicate-argument Semantics": (323, 342),
    "Distributional Semantics": (343, 368),
    "Reference Resolution": (369, 396),
    "Discourse": (397, 418),
    "Informtion Extraction": (421, 448),
    "Machine Translation": (449, 474),
    "Text Generation": (475, 491)
}
folder = "Natural Language Processing"

split_pdf(input_pdf, folder, config)