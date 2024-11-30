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
input_pdf = "Philippe_Jorion_-_Value_at_Risk_-_The_Ne.pdf"
config = {
    "Computing VaR": (109, 139),
    "Backtesting VaR": (143, 162),
    "Portfolio Risk": (163, 185),
    "Multivariate Models": (193, 216),
    "Forecasting Risk and Correlations": (223, 249),
    "VaR Methods": (251, 275),
    "VaR Mapping": (281, 306),
    "Monte Carlo Methods": (311, 335),
    "Liquid Risk": (337, 360),
    "Stress Testing": (361, 379)
}
folder = "Risk Analysis"

for key, value in config.items():
    config[key] = (value[0]+12, value[1]+12)

split_pdf(input_pdf, folder, config)