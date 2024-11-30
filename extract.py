import json

def extract_apis(swagger_content):
    data = json.loads(swagger_content)
    apis = []

    for path, methods in data.get('paths', {}).items():
        for method, details in methods.items():
            api = {
                'path': path,
                'method': method.upper(),
                'summary': details.get('summary', ''),
                'operationId': details.get('operationId', ''),
                'tags': details.get('tags', [])
            }
            apis.append(api)

    return apis

APIS = {}

def generate_api_json(api):
    if api.get('method') != 'GET':
        return None

    # Generate key from operationId or path
    key = api['operationId'].lower() if api['operationId'] else api['path'].split('/')[-1].lower()
    
    # Generate master_api from operationId or summary
    master_api = api['operationId'].lower() if api['operationId'] else api['summary'].lower().replace(' ', '_')
    
    # Generate target_table
    target_table = f"sinacor_{key}"

    APIS[master_api] = f"/Faturamento{api['path']}"

    json_output = {
        key: {
            "source": "api",
            "master_api": master_api,
            "target_table": target_table,
            "target_schema": "bronze_api_faturamento",
            "paginated": False,
            "id_columns": [],
            "transformations": [],
            "timestamp_columns": [],
            "pii_columns": [],
            "partition_columns": [],
            "modes": {
                "upsert": {},
                "overwrite": {}
            },
            "details": []
        }
    }

    return json_output

import json

if __name__ == '__main__':
    filename = 'apis.json'

    with open(filename) as f:
        swagger_content = f.read()
        apis = extract_apis(swagger_content)

    bronze_jsons = []

    for api in apis:
        bronze_json = generate_api_json(api)

        if bronze_json:
            bronze_jsons.append(bronze_json)

    filename = 'CONF_Segments_Faturamento.txt'
    
    with open(filename, 'w') as file:
        string = "CONF_Segments_Faturamento = {}\n\n"

        for bronze_json in bronze_jsons:
            string += "CONF_Segments_Faturamento.extend("
            string += json.dumps(bronze_json, indent=4)
            string += ")\n\n"

        file.write(string)

    filename = 'CONF_APIS.txt'

    with open(filename, 'w') as file:
        string = "APIS.extend({\n"

        for key, value in APIS.items():
            string += f'\t"{key}": "{value}",\n'

        string += "})"

        file.write(string)

    workflows = {
        "ETL_SINACOR_SINACOR_API_FATURAMENTO_FULL_LOAD": {
            "schedule": {
                "quartz_cron_expression": "0 0 9 ? * MON-FRI *",
                "timezone_id": "America/Sao_Paulo",
                "pause_status": "PAUSED"
            },
            "tasks": {}
        }
    }

    for bronze_json in bronze_jsons[:98]:
        key = list(bronze_json.keys())[0]
        workflows["ETL_SINACOR_SINACOR_API_FATURAMENTO_FULL_LOAD"]["tasks"][key] = {
            "segment": key,
            "layers": ["faturamento"],
            "mode": "overwrite",
            "is_full_load": "true",
            "version": "004"
        }
    
    filename = "WORKFLOWS.txt"

    with open(filename, 'w') as file:
        string = "WORKFLOWS.extend(\n"
        string += json.dumps(workflows, indent=4)
        string += ")"

        file.write(string)