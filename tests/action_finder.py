import cv2
import numpy as np
import os
import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

# --- CONFIGURAÇÕES ---

# Duração mínima para um clipe de ação ser considerado (em segundos)
MIN_ACTION_CLIP_DURATION = 2.0

def analyze_video_for_action(filepath):
    """
    Analisa o vídeo para encontrar momentos de alta ação usando fluxo óptico.
    Retorna uma lista de tuplas (timestamp, action_score).
    """
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise IOError(f"Não foi possível abrir o vídeo {filepath}")
    except Exception as e:
        print(f"\nErro ao abrir o vídeo {filepath}: {e}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Não foi possível ler o primeiro frame de {filepath}.")
        cap.release()
        return None, None
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    action_scores = []
    
    desc = f"Analisando Ação em {os.path.basename(filepath)}"
    frame_iterator = tqdm(range(1, total_frames), total=total_frames, desc=desc, unit="frame", leave=False)

    for frame_num in frame_iterator:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- MELHORIA: Análise de Ação Localizada ---
        # Em vez de analisar a imagem inteira, focamos em uma região central onde
        # ultrapassagens e interações são mais prováveis de ocorrer.
        h, w = gray.shape
        roi_y_start, roi_y_end = int(h * 0.1), int(h * 0.9)
        roi_x_start, roi_x_end = int(w * 0.1), int(w * 0.9)
        roi_prev_gray = prev_gray[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_gray = gray[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        flow = cv2.calcOpticalFlowFarneback(roi_prev_gray, roi_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Em vez da média, usamos o 95º percentil da magnitude do movimento.
        # Isso torna a detecção sensível a movimentos rápidos e localizados (como uma ultrapassagem),
        # ignorando o fato de que o resto da cena pode estar parado.
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        action_score = np.percentile(magnitude, 95)
        
        timestamp = frame_num / fps
        action_scores.append((timestamp, action_score))
        
        prev_gray = gray

    cap.release()
    return action_scores, fps

def get_best_moments(original_clip, action_scores, percentile_threshold):
    """Seleciona os melhores momentos com base na pontuação de ação."""
    if not action_scores:
        return []

    scores = [score for _, score in action_scores]
    # Define o limiar como um percentil das pontuações (ex: 80% mais altas)
    percentile_value = percentile_threshold * 100
    action_threshold = np.percentile(scores, percentile_value)
    
    tqdm.write(f"    -> Limiar de ação definido em: {action_threshold:.2f} (Top {(1-percentile_threshold)*100:.0f}%)")

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

    # Verifica se o último segmento vai até o final do vídeo
    if current_start is not None:
        duration = original_clip.duration - current_start
        if duration >= MIN_ACTION_CLIP_DURATION:
            good_segments.append((current_start, original_clip.duration))

    if not good_segments:
        tqdm.write(f"    -> Nenhum trecho de ação encontrado acima do limiar.")
        return []
        
    return [original_clip.subclip(start, end) for start, end in good_segments]

def main():
    parser = argparse.ArgumentParser(description="Seleciona os melhores momentos de um vídeo com base na ação.")
    parser.add_argument('file', help='O arquivo de vídeo para processar (idealmente o resultado do steady_cut.py).')
    parser.add_argument('-o', '--output', default='melhores_momentos.mp4', help='Nome do arquivo de vídeo de saída.')
    parser.add_argument(
        '--top',
        type=float,
        default=20.0,
        metavar='P',
        help='Percentual dos melhores momentos a serem selecionados (ex: 10 para os 10%% mais movimentados). Padrão: 20'
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Erro: O arquivo '{args.file}' não foi encontrado.")
        return

    action_scores, fps = analyze_video_for_action(args.file)
    
    if action_scores:
        # Converte o percentual do topo (ex: 20) para o limiar de percentil (ex: 0.8)
        percentile_threshold = (100 - args.top) / 100.0
        original_clip = VideoFileClip(args.file)
        best_clips = get_best_moments(original_clip, action_scores, percentile_threshold)
        
        if best_clips:
            print(f"\nCompilando {len(best_clips)} melhores momentos...")
            final_video = None
            try:
                final_video = concatenate_videoclips(best_clips, method="compose")
                final_video.write_videofile(args.output, codec="libx264", audio_codec="aac", threads=4, preset="medium")
                print(f"\nSucesso! Melhores momentos salvos como: {args.output}")
            finally:
                # Garante que todos os clipes (incluindo subclips) sejam fechados
                if final_video:
                    final_video.close()
                for clip in best_clips:
                    clip.close()
                original_clip.close()
        else:
            print("\nNão foram encontrados clipes de ação longos o suficiente para criar um vídeo.")
            original_clip.close()

if __name__ == "__main__":
    main()