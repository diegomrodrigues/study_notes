import json
from jsonschema import validate

def validar_json_com_schema(schema_file, json_file):
    """Valida um arquivo JSON com base em um schema JSON.

    Args:
        schema_file: Caminho para o arquivo JSON do schema.
        json_file: Caminho para o arquivo JSON a ser validado.

    Returns:
        True se o JSON for válido de acordo com o schema, False caso contrário.
        Imprime mensagens de erro detalhadas em caso de falha na validação.
    """
    try:
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        validate(instance=json_data, schema=schema)
        return True

    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}")
        return False
    except Exception as e:
        print(f"Erro de validação: {e}")
        return False

# Exemplo de uso:
schema_file = "schema.json"  # Substitua pelo caminho do seu arquivo de schema
json_file = "example.json"  # Substitua pelo caminho do seu arquivo JSON

if validar_json_com_schema(schema_file, json_file):
    print("O JSON é válido de acordo com o schema.")
else:
    print("O JSON não é válido de acordo com o schema.") 