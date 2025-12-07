# main.py
import os
from google.cloud import storage
from video_processor import generate_highlight_video
import logging

# Configura o logger para que as mensagens apareçam nos logs do Cloud Functions
# com um formato claro.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- Configuração do Ambiente GCP ---
# Estes nomes de bucket devem ser criados no seu projeto GCP
# Ex: my-awesome-video-uploads, my-awesome-video-results
RESULTS_BUCKET_NAME = os.environ.get('RESULTS_BUCKET_NAME')

# Limite de tamanho de arquivo em bytes para processamento nesta função (ex: 1 GB)
# Arquivos maiores podem precisar de uma arquitetura diferente (ex: Cloud Run, GKE).
MAX_FILE_SIZE_BYTES = int(os.environ.get('MAX_FILE_SIZE_BYTES', 1 * 1024 * 1024 * 1024))

# Inicializa o cliente do Cloud Storage
storage_client = storage.Client()

def process_video_from_gcs(event, context):
    """
    Cloud Function acionada por um upload de arquivo para o GCS.
    :param event: O dicionário com metadados do evento.
    :param context: O objeto com metadados da execução da função.
    """
    
    # Pega informações do arquivo que acionou o evento
    file_data = event
    bucket_name = file_data['bucket']
    file_name = file_data['name']
    file_size = int(file_data.get('size', 0))

    if file_size > MAX_FILE_SIZE_BYTES:
        logging.error(f"Arquivo {file_name} ({file_size / (1024**3):.2f} GB) excede o limite de {MAX_FILE_SIZE_BYTES / (1024**3):.2f} GB para esta função. Abortando.")
        # Opcional: Mover o arquivo para um bucket de "quarentena" ou acionar outro processo.
        return
    
    logging.info(f"Iniciando função para o arquivo: {file_name} do bucket: {bucket_name}.")

    base_name = os.path.basename(file_name)
    # Define os caminhos de arquivo no ambiente temporário da função
    # O diretório /tmp é o único local gravável em um Cloud Function
    temp_input_path = f"/tmp/{base_name}"
    temp_steady_output_path = f"/tmp/steady_{base_name}"
    temp_output_path = f"/tmp/highlight_{base_name}"

    try:
        # --- Download do Vídeo ---
        source_bucket = storage_client.bucket(bucket_name)
        blob = source_bucket.blob(file_name)
        
        logging.info(f"Baixando {file_name} para {temp_input_path}...")
        blob.download_to_filename(temp_input_path)
        logging.info("Download concluído.")

        # --- Processamento do Vídeo ---
        # Chama a lógica principal que refatoramos
        # Você pode pegar os parâmetros (blur, top_percent) dos metadados do arquivo se quiser
        success = generate_highlight_video(
            input_path=temp_input_path,
            output_path=temp_output_path,
            steady_output_path=temp_steady_output_path,
            blur_threshold=60,  # Pode ser configurado via variáveis de ambiente
            top_percent=20
        )

        # --- Upload do Resultado ---
        if success:
            logging.info(f"Processamento bem-sucedido. Fazendo upload do resultado para o bucket {RESULTS_BUCKET_NAME}...")
            if not RESULTS_BUCKET_NAME:
                logging.error("Nome do bucket de resultados (RESULTS_BUCKET_NAME) não configurado. Abortando upload.")
                return

            results_bucket = storage_client.bucket(RESULTS_BUCKET_NAME)

            # Upload do vídeo de ação (highlight)
            action_blob_name = f"highlight_{base_name}"
            output_blob = results_bucket.blob(action_blob_name)
            output_blob.upload_from_filename(temp_output_path)
            logging.info(f"Upload do vídeo de ação concluído. Arquivo disponível em gs://{RESULTS_BUCKET_NAME}/{action_blob_name}")
            
            # Upload do vídeo estabilizado
            if os.path.exists(temp_steady_output_path):
                steady_blob_name = f"steady_{base_name}"
                steady_blob = results_bucket.blob(steady_blob_name)
                steady_blob.upload_from_filename(temp_steady_output_path)
                logging.info(f"Upload do vídeo estável concluído. Arquivo disponível em gs://{RESULTS_BUCKET_NAME}/{steady_blob_name}")

        else:
            logging.warning("Processamento falhou ou não gerou vídeo. Nenhum arquivo será enviado.")

    except Exception as e:
        logging.critical(f"Ocorreu um erro não tratado durante o processamento de {file_name}: {e}", exc_info=True)

    finally:
        # --- Limpeza ---
        # Garante que os arquivos temporários sejam removidos mesmo se ocorrer um erro.
        logging.info("Iniciando limpeza dos arquivos temporários...")
        for path in [temp_input_path, temp_output_path, temp_steady_output_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logging.info(f"Arquivo temporário {path} removido.")
                except OSError as e:
                    logging.error(f"Erro ao remover arquivo temporário {path}: {e}")
        logging.info("Limpeza concluída. Função finalizada.")
