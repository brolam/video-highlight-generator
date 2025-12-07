import cv2
import os
import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips

from tqdm import tqdm
# --- CONFIGURAÇÕES ---

# Duração mínima para um clipe ser considerado (em segundos)
# Evita que o vídeo final fique piscando com cortes de milissegundos
MIN_CLIP_DURATION = 1.0 

def calculate_sharpness(image):
    """Calcula a variância do Laplaciano. Retorna alto para imagem nítida, baixo para borrada."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_video(filepath, blur_threshold):
    """Analisa um vídeo e retorna uma lista de subclips estáveis."""
    try:
        original_clip = VideoFileClip(filepath)
    except Exception as e:
        print(f"\nErro ao abrir o vídeo {filepath}: {e}")
        return []

    fps = original_clip.fps
    if not fps or fps == 0:
        print(f"\nNão foi possível determinar o FPS de {filepath}. Pulando.")
        original_clip.close()
        return []
    
    good_segments = []
    current_start = None
    
    # Usamos tqdm para uma barra de progresso
    desc = f"Analisando {os.path.basename(filepath)}"
    frame_iterator = original_clip.iter_frames(fps=fps, with_times=True)
    total_frames = int(original_clip.duration * fps)

    for timestamp, frame in tqdm(frame_iterator, total=total_frames, desc=desc, unit="frame", leave=False):
        sharpness = calculate_sharpness(frame)
        
        if sharpness > blur_threshold:
            if current_start is None:
                current_start = timestamp
        else:
            if current_start is not None:
                duration = timestamp - current_start
                if duration >= MIN_CLIP_DURATION:
                    good_segments.append((current_start, timestamp))
                current_start = None
    
    if not good_segments:
        tqdm.write(f"    -> Nenhum trecho estável encontrado em {os.path.basename(filepath)}.")
        original_clip.close()
        return []
        
    subclips = [original_clip.subclip(start, end) for start, end in good_segments]
    # Não fechamos o 'original_clip' aqui, pois os subclips dependem dele.
    # MoviePy gerencia isso ao concatenar.
    return subclips

def main():
    """Função principal para processar vídeos passados como argumentos."""
    parser = argparse.ArgumentParser(
        description="Edita vídeos de ciclismo, mantendo apenas os trechos nítidos e estáveis.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('files', nargs='+', help='Um ou mais arquivos de vídeo para processar.')
    parser.add_argument(
        '--blur', 
        type=int, 
        default=60, 
        help="""Threshold de nitidez.
Valor menor (ex: 50) = Aceita mais imagens (mesmo um pouco tremidas).
Valor maior (ex: 300) = Muito exigente (só imagens cristalinas).
Padrão: 60"""
    )
    parser.add_argument('-o', '--output', default='video_final_ciclismo.mp4', help='Nome do arquivo de vídeo de saída. Padrão: video_final_ciclismo.mp4')

    args = parser.parse_args()

    all_clips = []
    print(f"Encontrados {len(args.files)} vídeos. Iniciando processamento...")
    print(f"Threshold de Nitidez: {args.blur} | Duração Mínima do Clipe: {MIN_CLIP_DURATION}s\n")
    
    for filename in args.files:
        if filename == args.output: continue # Pula o arquivo de saída se já existir
        
        if not os.path.exists(filename):
            print(f"Aviso: O arquivo '{filename}' não foi encontrado e será ignorado.")
            continue
        clips = process_video(filename, args.blur)
        all_clips.extend(clips)
        
    if all_clips:
        print(f"\nCompilando {len(all_clips)} trechos selecionados...")
        # Usamos 'method="compose"' para melhor gerenciamento de memória
        final_video = concatenate_videoclips(all_clips, method="compose")
        final_video.write_videofile(args.output, codec="libx264", audio_codec="aac", threads=4, preset="medium")
        for clip in all_clips: # Liberar recursos explicitamente
            clip.close()
        print(f"\nSucesso! Vídeo salvo como: {args.output}")
    else:
        print(f"\nNenhum trecho com qualidade suficiente foi encontrado. Tente usar um valor de --blur menor que {args.blur}.")

if __name__ == "__main__":
    main()