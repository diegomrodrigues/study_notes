```python
import asyncio
import aiohttp
import ijson
from datetime import datetime
from typing import AsyncGenerator, Dict, Any
import logging
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AllocationStreamGenerator:
    def __init__(self, api_url: str, poll_interval: float = 1.0):
        """
        Inicializa o gerador de stream de alocações.
        
        Args:
            api_url: URL da API que retorna as alocações
            poll_interval: Intervalo em segundos entre as consultas à API
        """
        self.api_url = api_url
        self.poll_interval = poll_interval
        self.last_processed_time = None
        self._session = None
    
    async def __aenter__(self):
        """Contexto assíncrono para gerenciar a sessão HTTP."""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Fecha a sessão HTTP ao finalizar."""
        if self._session:
            await self._session.close()
    
    def _parse_horario_alteracao(self, horario: str) -> datetime:
        """Converte o horário de alteração para objeto datetime."""
        return datetime.strptime(horario, "%Y/%m/%d %H:%M:%S")
    
    async def _stream_allocations(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Faz streaming das alocações usando ijson, começando diretamente 
        da chave listaAlocacao.
        """
        try:
            headers = {
              'COMPANY_ID': '1130',
              'SISTEMA_ORIGEM': 'Datalake',
              'Authorization': 'Bearer yP8JD6BJj5BSdXRAOlPCWxRIf-bf1gE20qfAbNvraiOH_kUU9NcqIqXPmx8MIKT7-q9I4n2s-DBTXxZPHp8MPbbVZpOq067nPV74c1UAjhBav8emFiJhiD9l5-NdJRAzVPpPcZqLmfXCPwiTOENSpwVnvKxa6EWIK7joj1LGfffsTjRFefGK1wtjxVnUgEIZUK23kLUBG1ixn-NVGmPbUpnUJQaPRM0cKb6VDXCYosnszomPxblq7tNTrcTJS6tR22T6VMrHlxfMRLYzA0WbjNn1-TRiVlF_64gCgR2Epx7dqN2n49a2cWMuVWm_NsZnGmu7DHselBqAwowoRTW-6A'
            }

            async with self._session.get(self.api_url, headers=headers) as response:
                response.raise_for_status()
                
                current_item = {}
                inside_lista_alocacao = False
                
                # Stream do corpo da resposta
                parser = ijson.parse(await response.read())
                
                for prefix, event, value in parser:
                    # Se encontrou o início da lista de alocações
                    if prefix == 'listaAlocacao' and event == 'start_array':
                        inside_lista_alocacao = True
                        continue
                        
                    # Se ainda não chegou na lista, continua procurando
                    if not inside_lista_alocacao:
                        continue
                        
                    # Se encontrou o fim da lista, encerra
                    if prefix == 'listaAlocacao' and event == 'end_array':
                        break
                    
                    # Processa os campos de cada item da lista
                    if inside_lista_alocacao:
                        if event == 'start_map':  # Início de um novo item
                            current_item = {}
                        elif event == 'end_map':  # Fim do item atual
                            if current_item:  # Se tem um item completo
                                current_time = self._parse_horario_alteracao(
                                    current_item["horarioAlteracao"]
                                )
                                
                                # Se é novo ou primeira execução
                                if (not self.last_processed_time or 
                                    current_time > self.last_processed_time):
                                    self.last_processed_time = current_time
                                    yield current_item
                        elif event != 'map_key':  # Valores dos campos
                            # O prefix vai ter o formato "listaAlocacao.item.campo"
                            field = prefix.split('.')[-1]
                            current_item[field] = value
                            
        except aiohttp.ClientError as e:
            logger.error(f"Erro ao buscar alocações: {e}")
    
    async def stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Gera um stream contínuo de novas alocações.
        
        Yields:
            Dict[str, Any]: Nova alocação detectada
        """
        if not self._session:
            raise RuntimeError("Gerador deve ser usado como context manager")
        
        while True:
            async for allocation in self._stream_allocations():
                yield allocation
            
            await asyncio.sleep(self.poll_interval)
            print("Começando novo request")

async def process_allocation(allocation: Dict[str, Any]):
    """
    Função exemplo para processar cada alocação.
    Adapte conforme suas necessidades.
    """
    print(f"Processando alocação: {allocation['numeroSequencialAlocacao']}")
    print(f"Horário: {allocation['horarioAlteracao']}")
    
    # Exemplo de processamento baseado no status
    if allocation["statusAlocacao"] == 1:
        print("Alocação ativa processada")

async def main():
    api_url = "http://10.41.12.151:25149/AlocacaoUnificada/api/Alocacoes?sistemaOrigem=DATALAKE&request.codigoIdentificadorOperacao=A&request.dataRealizacaoNegocio=2024-11-08&request.horarioAlteracao=2024/11/08 10:15:13"
    
    async with AllocationStreamGenerator(api_url) as generator:
        async for allocation in generator.stream():
            await process_allocation(allocation)

if __name__ == "__main__":
    await main()
```

