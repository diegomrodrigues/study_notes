import os
import pandas as pd
import unicodedata
import re
import sys

class StudiesNotesHandler:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = {}
        self.load_existing_data()

    def load_existing_data(self):
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                main_folder = item
                md_file = os.path.join(item_path, 'studies.md')
                csv_file = os.path.join(item_path, 'studies.csv')
                if os.path.exists(md_file):
                    self.data[main_folder] = self.parse_markdown_table(md_file)
                elif os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        if df.empty:
                            print(f"Warning: CSV file for {main_folder} is empty. Initializing with empty data.")
                            self.data[main_folder] = []
                        else:
                            self.data[main_folder] = df.to_dict('records')
                    except pd.errors.EmptyDataError:
                        print(f"Warning: CSV file for {main_folder} is empty. Initializing with empty data.")
                        self.data[main_folder] = []
                else:
                    self.data[main_folder] = []

    def parse_markdown_table(self, md_file):
        try:
            df = pd.read_table(md_file, sep="|", header=0, skiprows=2, skipinitialspace=True) \
                .dropna(axis=1, how='all') \
                .iloc[1:]  \
                
            # Remover espaços dos nomes das colunas
            df.columns = df.columns.str.strip()

            # Remover espaços dos valores em todas as colunas de string
            df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

            return df.to_dict('records')
        except Exception as e:
            return []
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        table_content = re.search(r'\|(.+\|)+\n(\|[-:]+\|)+\n((\|.+\|)+\n)+', content)
        if table_content:
            table_lines = table_content.group().split('\n')[1:-1]
            headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]
            data = []
            for line in table_lines[2:]:
                values = [v.strip() for v in line.split('|')[1:-1]]
                data.append(dict(zip(headers, values)))
            return data
        return []

    def normalize_filename(self, filename):
        return unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')

    def update_data(self, file_path):
        rel_path = os.path.relpath(file_path, self.root_dir)
        parts = rel_path.split(os.sep)
        
        if len(parts) < 2:
            return  # Ignora arquivos na raiz

        main_folder = parts[0]
        if len(parts) == 2:
            folder = ""
            subfolder = ""
            filename = parts[1]
        elif len(parts) == 3:
            folder = parts[1]
            subfolder = ""
            filename = parts[2]
        else:
            folder = parts[1]
            subfolder = os.sep.join(parts[2:-1])
            filename = parts[-1]

        normalized_filename = self.normalize_filename(filename)

        if main_folder not in self.data:
            self.data[main_folder] = []

        existing_entry = next((item for item in self.data[main_folder] 
                               if item.get('Folder') == folder 
                               and item.get('Subfolder') == subfolder 
                               and item.get('Filename') == normalized_filename), None)

        if not existing_entry:
            self.data[main_folder].append({
                'Folder': folder,
                'Subfolder': subfolder,
                'Filename': normalized_filename,
                'Was Studied?': 'No',
                'Was Printed?': 'No',
                'Was Reviewed?': 'No'
            })

    def remove_deleted_files(self):
        for main_folder in self.data:
            existing_files = set()
            main_folder_path = os.path.join(self.root_dir, main_folder)
            for root, _, files in os.walk(main_folder_path):
                for file in files:
                    if file.endswith('.md') and file != 'studies.md':
                        rel_path = os.path.relpath(os.path.join(root, file), main_folder_path)
                        parts = rel_path.split(os.sep)
                        if len(parts) == 1:
                            folder = ""
                            subfolder = ""
                            filename = parts[0]
                        elif len(parts) == 2:
                            folder = parts[0]
                            subfolder = ""
                            filename = parts[1]
                        else:
                            folder = parts[0]
                            subfolder = os.sep.join(parts[1:-1])
                            filename = parts[-1]
                        filename = self.normalize_filename(filename)
                        existing_files.add((folder, subfolder, filename))
            
            self.data[main_folder] = [item for item in self.data[main_folder]
                                      if (item['Folder'], item['Subfolder'], item['Filename']) in existing_files]

    def save_data(self, main_folder):
        df = pd.DataFrame(self.data[main_folder])
        
        sort_columns = ['Folder', 'Subfolder', 'Filename']
        existing_columns = [col for col in sort_columns if col in df.columns]
        
        if existing_columns:
            df.sort_values(existing_columns, inplace=True)
        
        main_folder_path = os.path.join(self.root_dir, main_folder)
        os.makedirs(main_folder_path, exist_ok=True)
        
        csv_file = os.path.join(main_folder_path, 'studies.csv')
        df.to_csv(csv_file, index=False)
        
        md_file = os.path.join(main_folder_path, 'studies.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# {main_folder} Studies ✅\n\n")
            f.write(df.to_markdown(index=False))

    def scan_existing_files(self):
        for root, _, files in os.walk(self.root_dir):
            if '.git' in root.split(os.sep):
                continue
            for file in files:
                if file.endswith('.md') and file != 'studies.md':
                    file_path = os.path.join(root, file)
                    self.update_data(file_path)

    def process_all(self):
        self.scan_existing_files()
        self.remove_deleted_files()
        for main_folder in self.data:
            self.save_data(main_folder)

def main():
    root_dir = '.'

    handler = StudiesNotesHandler(root_dir)
    handler.process_all()
    print("Processing complete. Check the 'studies.md' and 'studies.csv' files in each main folder.")

if __name__ == "__main__":
    main()