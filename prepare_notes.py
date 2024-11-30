import json
import os
from pathlib import Path

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_files(topics, template, output_folder):
    for idx, topic in enumerate(topics["topics"]):
        filename = f"Untitled{idx or ''}.md"
        
        output_path = output_folder / filename
        with open(output_path, 'w', encoding='utf-8') as file:
            chunks = "\n".join(topic["chunks"])
            file.write(f"X = {topic['subtopic']} \n\n {chunks}")
        print(f"Generated: {output_path}")

def main():
    # Use Path for better cross-platform compatibility
    input_topics = Path("Time Series Analysis") / "Differential Equations" / "topics.json"
    output_folder = Path("Time Series Analysis") / "Differential Equations"
    input_template = Path("Prompts") / "gpt-o-preview" / "Resumos.md"

    # Ensure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load data and template
    topics = load_json(input_topics)
    template = read_template(input_template)

    # Generate files
    generate_files(topics, template, output_folder)

if __name__ == "__main__":
    main()