# video_processor.py
import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm
import logging

# --- CONFIGURAÇÕES ---
MIN_CLIP_DURATION_STEADY = 1.0
MIN_ACTION_CLIP_DURATION = 2.0

# --- FUNÇÕES DE STEADY_CUT.PY ---

def _calculate_sharpness(image):
    """Calcula a variância do Laplaciano."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _get_steady_clips(original_clip, blur_threshold):
    """Analisa um vídeo e retorna uma lista de subclips estáveis."""
    fps = original_clip.fps
    if not fps or fps == 0:
        logging.warning("Não foi possível determinar o FPS. Pulando análise de estabilidade.")
        return [original_clip] # Retorna o clipe original se não puder analisar

    good_segments = []
    current_start = None
    
    total_frames = int(original_clip.duration * fps)
    frame_iterator = tqdm(original_clip.iter_frames(fps=fps, with_times=True), total=total_frames, desc="1/2: Analisando Estabilidade", unit="frame", disable=None)

    for timestamp, frame in frame_iterator:
        sharpness = _calculate_sharpness(frame)
        
        if sharpness > blur_threshold:
            if current_start is None:
                current_start = timestamp
        else:
            if current_start is not None:
                duration = timestamp - current_start
                if duration >= MIN_CLIP_DURATION_STEADY:
                    good_segments.append((current_start, timestamp))
                current_start = None
    
    if not good_segments:
        logging.warning("Nenhum trecho estável encontrado. O vídeo de ação pode ficar vazio.")
        return []
        
    return [original_clip.subclip(start, end) for start, end in good_segments]

# --- FUNÇÕES DE ACTION_FINDER.PY ---

def _analyze_video_for_action(clip_concatenado):
    """Analisa o clipe concatenado para encontrar momentos de alta ação."""
    fps = clip_concatenado.fps
    action_scores = []
    
    # Usar iter_frames de moviepy para evitar reabrir o vídeo com OpenCV
    frame_iterator = clip_concatenado.iter_frames(fps=fps)
    
    # Ler o primeiro frame
    try:
        prev_frame = next(frame_iterator)
    except StopIteration:
        return [], fps # Vídeo vazio ou de um frame

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    total_frames = int(clip_concatenado.duration * fps)
    progress_bar = tqdm(frame_iterator, total=total_frames-1, desc="2/2: Analisando Ação", unit="frame", disable=None)

    for i, frame in enumerate(progress_bar):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = np.mean(magnitude)
        
        # O timestamp é relativo ao início do clipe concatenado
        timestamp = (i + 1) / fps
        action_scores.append((timestamp, mean_magnitude))
        
        prev_gray = gray

    return action_scores, fps

def _get_best_moments(original_clip, action_scores, percentile_threshold):
    """Seleciona os melhores momentos com base na pontuação de ação."""
    if not action_scores:
        return []

    scores = [score for _, score in action_scores]
    action_threshold = np.percentile(scores, percentile_threshold * 100)
    
    logging.info(f"Limiar de ação definido em: {action_threshold:.2f} (Top {(1-percentile_threshold)*100:.0f}%)")

    good_segments = []
    current_start = None
    
    for timestamp, score in action_scores:
        if score > action_threshold:
            if current_start is None:
                current_start = timestamp
        else:
            if current_start is not None:
                duration = timestamp - current_start
                if duration >= MIN_ACTION_CLIP_DURATION:
                    good_segments.append((current_start, timestamp))
                current_start = None

    if current_start is not None:
        duration = original_clip.duration - current_start
        if duration >= MIN_ACTION_CLIP_DURATION:
            good_segments.append((current_start, original_clip.duration))

    if not good_segments:
        logging.warning("Nenhum trecho de ação encontrado acima do limiar.")
        return []
        
    return [original_clip.subclip(start, end) for start, end in good_segments]

# --- FUNÇÃO PRINCIPAL DO PIPELINE ---

def generate_highlight_video(input_path, output_path, blur_threshold=60, top_percent=20):
    """
    Pipeline completo: carrega um vídeo, remove partes tremidas e extrai os melhores momentos de ação.
    """
    logging.info(f"Iniciando processamento de {input_path}")
    
    try:
        original_clip = VideoFileClip(input_path)
    except Exception as e:
        logging.error(f"Erro ao abrir o vídeo {input_path}: {e}")
        return False

    # 1. Remover trechos tremidos/borrados
    steady_clips = _get_steady_clips(original_clip, blur_threshold)
    
    if not steady_clips:
        logging.warning("Nenhum clipe estável encontrado. Abortando.")
        original_clip.close()
        return False

    # Concatena os clipes estáveis em um vídeo temporário em memória
    steady_video = concatenate_videoclips(steady_clips, method="compose")
    logging.info("Clipes estáveis concatenados.")

    # 2. Encontrar os momentos de maior ação no vídeo já estabilizado
    action_scores, _ = _analyze_video_for_action(steady_video)
    
    if not action_scores:
        logging.warning("Não foi possível analisar a ação no vídeo. Abortando.")
        steady_video.close()
        original_clip.close()
        return False

    percentile_threshold = (100 - top_percent) / 100.0
    best_clips = _get_best_moments(steady_video, action_scores, percentile_threshold)

    # 3. Compilar o vídeo final
    if best_clips:
        logging.info(f"Compilando {len(best_clips)} melhores momentos...")
        final_video = concatenate_videoclips(best_clips, method="compose")
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=4, preset="medium")
        
        # Liberar recursos
        final_video.close()
        for clip in best_clips:
            clip.close()
        success = True
    else:
        logging.warning("Nenhum clipe de ação longo o suficiente foi encontrado para criar um vídeo.")
        success = False

    # Liberar recursos restantes
    steady_video.close()
    original_clip.close()
    for clip in steady_clips:
        clip.close()
        
    return success
