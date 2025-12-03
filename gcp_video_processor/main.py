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
    
    logging.info(f"Iniciando função para o arquivo: {file_name} do bucket: {bucket_name}.")

    # Define os caminhos de arquivo no ambiente temporário da função
    # O diretório /tmp é o único local gravável em um Cloud Function
    temp_input_path = f"/tmp/{os.path.basename(file_name)}"
    temp_output_path = f"/tmp/highlight_{os.path.basename(file_name)}"

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
        blur_threshold=60,  # Pode ser configurado via variáveis de ambiente
        top_percent=20
    )

    # --- Upload do Resultado ---
    if success:
        logging.info(f"Processamento bem-sucedido. Fazendo upload do resultado para o bucket {RESULTS_BUCKET_NAME}...")
        results_bucket = storage_client.bucket(RESULTS_BUCKET_NAME)
        output_blob = results_bucket.blob(f"highlight_{os.path.basename(file_name)}")
        
        output_blob.upload_from_filename(temp_output_path)
        logging.info(f"Upload concluído. Arquivo disponível em gs://{RESULTS_BUCKET_NAME}/{output_blob.name}")
    else:
        logging.warning("Processamento falhou ou não gerou vídeo. Nenhum arquivo será enviado.")

    # --- Limpeza ---
    # Remove os arquivos temporários para liberar espaço
    if os.path.exists(temp_input_path):
        os.remove(temp_input_path)
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
        
    logging.info("Limpeza concluída. Função finalizada.")
