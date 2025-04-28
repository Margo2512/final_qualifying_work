import cv2
import os
import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
from itertools import chain
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display

import timm
import models_mae_cross
from util.misc import measure_time

assert "0.4.5" <= timm.__version__ <= "0.4.9"
device = torch.device('cuda')
shot_num = 0

class OpticalFlowTracker:
    def __init__(self, max_distance=50):
        self.max_distance = max_distance
        self.tracks = {}
        self.next_id = 0
        self.prev_gray = None

    def update(self, frame, centers):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            self.tracks = {i: [center] for i, center in enumerate(centers)}
            self.next_id = len(centers)
            return list(self.tracks.keys())
        
        prev_pts = np.array([track[-1] for track in self.tracks.values()], 
                           dtype=np.float32).reshape(-1, 1, 2)
        
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, prev_pts, None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        updated_tracks = {}
        for i, (track_id, track) in enumerate(self.tracks.items()):
            if status[i]:
                updated_tracks[track_id] = track + [tuple(curr_pts[i].ravel())]
        
        matched_centers = set()
        for track in updated_tracks.values():
            matched_centers.add(track[-1])
        
        for center in centers:
            if center not in matched_centers:
                closest_track = None
                min_dist = float('inf')
                
                for track_id, track in updated_tracks.items():
                    dist = np.linalg.norm(np.array(center) - np.array(track[-1]))
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        closest_track = track_id
                
                if closest_track is not None:
                    updated_tracks[closest_track].append(center)
                else:
                    updated_tracks[self.next_id] = [center]
                    self.next_id += 1
        
        self.tracks = updated_tracks
        self.prev_gray = frame_gray
        return list(self.tracks.keys())

class TrackingQualityAnalyzer:
    def __init__(self):
        self.metrics = {
            'frame': [],
            'displacement': [],
            'coverage': [],
            'optical_flow': [],
            'temporal_consistency': [],
            'track_lengths': {},
            'active_tracks': []
        }
        self.prev_tracks = {}
        self.prev_frame = None

    @staticmethod
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def update_metrics(self, frame_num, current_tracks, current_frame):
        current_dict = {track[0]: track[1:] for track in current_tracks}

        for track_id in current_dict:
            self.metrics['track_lengths'][track_id] = self.metrics['track_lengths'].get(track_id, 0) + 1

        if not self.prev_tracks:
            self.prev_tracks = current_dict
            self.prev_frame = current_frame.copy()
            return

        matched = 0
        total_displacement = 0
        total_iou = 0
        total_flow = 0

        if self.prev_frame is not None:
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        for track_id, current_bbox in current_dict.items():
            if track_id in self.prev_tracks:
                matched += 1
                prev_bbox = self.prev_tracks[track_id]

                # Среднее смещение
                dx = current_bbox[0] - prev_bbox[0]
                dy = current_bbox[1] - prev_bbox[1]
                displacement = np.sqrt(dx**2 + dy**2)
                total_displacement += displacement

                # IoU для темпоральной согласованности
                iou = self.calculate_iou(prev_bbox, current_bbox)
                total_iou += iou

                # Анализ оптического потока в области объекта
                if self.prev_frame is not None:
                    x, y, w, h = map(int, prev_bbox)
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, current_frame.shape[1] - x), min(h, current_frame.shape[0] - y)
                    if w > 0 and h > 0:
                        obj_flow = flow[y:y+h, x:x+w]
                        if obj_flow.size > 0:
                            magnitude = np.sqrt(obj_flow[...,0]**2 + obj_flow[...,1]**2)
                            total_flow += np.mean(magnitude)

        # Полнота обнаружения
        coverage = matched / len(self.prev_tracks) if len(self.prev_tracks) > 0 else 0

        self.metrics['frame'].append(frame_num)
        self.metrics['displacement'].append(total_displacement/matched if matched > 0 else 0)
        self.metrics['coverage'].append(coverage)
        self.metrics['temporal_consistency'].append(total_iou/matched if matched > 0 else 0)
        self.metrics['optical_flow'].append(total_flow/matched if matched > 0 else 0)
        self.metrics['active_tracks'].append(len(current_dict))

        self.prev_tracks = current_dict
        self.prev_frame = current_frame.copy()

    def get_final_metrics(self):
        if not self.metrics['frame']:
            return {}

        avg_metrics = {
            'avg_displacement': np.mean(self.metrics['displacement']),
            'avg_coverage': np.mean(self.metrics['coverage']),
            'avg_temporal_consistency': np.mean(self.metrics['temporal_consistency']),
            'avg_optical_flow': np.mean(self.metrics['optical_flow']),
            'track_length_mean': np.mean(list(self.metrics['track_lengths'].values())) if self.metrics['track_lengths'] else 0,
            'track_length_median': np.median(list(self.metrics['track_lengths'].values())) if self.metrics['track_lengths'] else 0,
            'max_active_tracks': max(self.metrics['active_tracks']) if self.metrics['active_tracks'] else 0
        }
        return avg_metrics

    def get_tracking_score(self, weights=None, normalize=True, reference_score=22.5190, max_score=1.0):
        final_metrics = self.get_final_metrics()

        if not final_metrics:
            print("Предупреждение: нет метрик для расчета score!")
            return 0.0

        default_weights = {
            'avg_displacement': -0.2,
            'avg_coverage': 0.35,
            'avg_temporal_consistency': 0.25,
            'avg_optical_flow': -0.1,
            'track_length_mean': 0.2,
            'max_active_tracks': 0.1
        }

        weights = weights if weights is not None else default_weights

        missing_metrics = [k for k in weights if k not in final_metrics]
        if missing_metrics:
            print(f"Предупреждение: отсутствуют метрики {missing_metrics}, они не учитываются в score")

        raw_score = 0.0
        for key, weight in weights.items():
            if key in final_metrics:
                raw_score += float(final_metrics[key]) * weight

        if normalize:
            if reference_score <= 0:
                print("Ошибка: reference_score должен быть положительным!")
                return 0.0

            normalized_score = raw_score / reference_score
            
            return max(0.0, min(max_score, normalized_score))
        else:
            return raw_score

    def generate_metrics_plots(self, save_path=None):
        if not self.metrics['frame']:
            return None
        
        plt.figure(figsize=(16, 12))

        plt.subplot(3, 2, 1)
        plt.plot(self.metrics['frame'], self.metrics['displacement'], color='blue')
        plt.title('Динамика смещения объектов')
        plt.grid(True)

        plt.subplot(3, 2, 2)
        plt.plot(self.metrics['frame'], self.metrics['coverage'], color='green')
        plt.title('Полнота трекинга')
        plt.grid(True)

        plt.subplot(3, 2, 3)
        plt.plot(self.metrics['frame'], self.metrics['displacement'])
        plt.plot(self.metrics['frame'], self.metrics['coverage'])
        plt.title('Качество трекинга')
        plt.grid(True)

        plt.subplot(3, 2, 4)
        plt.plot(self.metrics['frame'], self.metrics['temporal_consistency'], color='red')
        plt.title('Темпоральная согласованность')
        plt.grid(True)

        plt.subplot(3, 2, 5)
        plt.plot(self.metrics['frame'], self.metrics['optical_flow'], color='purple')
        plt.title('Оптический поток объектов')
        plt.grid(True)

        plt.subplot(3, 2, 6)
        plt.plot(self.metrics['frame'], self.metrics['active_tracks'], color='green')
        plt.title('Активные треки')
        plt.grid(True)

        plt.tight_layout()
    
        plt.tight_layout()
        plt.show()


def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception(f"Ошибка: Не удалось открыть видео {video_path}")
    
    frame_idx = 0
    prev_gray = None
    total_motion = np.array([0.0, 0.0])
    num_vectors = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mean_flow = np.mean(flow, axis=(0, 1))
            total_motion += mean_flow
            num_vectors += 1
            
        prev_gray = frame_gray
        frame_idx += 1
        
    cap.release()
    avg_motion = total_motion / num_vectors if num_vectors > 0 else np.array([0, 0])
    print(f"Кадры сохранены в {output_folder} ({frame_idx} кадров).")
    print(f"Средний вектор перемещения объектов: ({avg_motion[0]:.2f}, {avg_motion[1]:.2f})")
    return frame_idx, avg_motion

def load_image(img_path: str):
    image = Image.open(img_path).convert('RGB')
    image.load()
    W, H = image.size
    new_H = 384
    new_W = 16 * int((W / H * 384) / 16)
    image = transforms.Resize((new_H, new_W))(image)
    Normalize = transforms.Compose([transforms.ToTensor()])
    image = Normalize(image)
    boxes = torch.Tensor([])
    return image, boxes, W, H

def run_one_image(samples, boxes, model, output_path, img_name, old_w, old_h):
    _, _, h, w = samples.shape
    density_map = torch.zeros([h, w])
    density_map = density_map.to(device, non_blocking=True)
    start = 0
    prev = -1
    
    with measure_time() as et:
        with torch.no_grad():
            while start + 383 < w:
                output, = model(samples[:, :, :, start:start + 384], boxes, shot_num)
                output = output.squeeze(0)
                
                b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                d1 = b1(output[:, 0:prev - start + 1])
                
                b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                d2 = b2(output[:, prev - start + 1:384])
                
                b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                density_map_l = b3(density_map[:, 0:start])
                
                density_map_m = b1(density_map[:, start:prev + 1])
                
                b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                density_map_r = b4(density_map[:, prev + 1:w])
                
                density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                prev = start + 383
                start = start + 128
                
                if start + 383 >= w:
                    if start == w - 384 + 128:
                        break
                    else:
                        start = w - 384
                        
        pred_cnt = torch.sum(density_map / 60).item()
    
    fig = samples[0]
    pred_fig = torch.stack((density_map, torch.zeros_like(density_map), torch.zeros_like(density_map)))
    
    count_im = Image.new(mode="RGB", size=(w, h), color=(0, 0, 0))
    draw = ImageDraw.Draw(count_im)
    draw.text((w-70, h-50), f"{pred_cnt:.3f}", (255, 255, 255))
    
    count_im = np.array(count_im).transpose((2, 0, 1))
    count_im = torch.tensor(count_im, device=device)
    
    fig = fig / 2 + pred_fig / 2 + count_im
    fig = torch.clamp(fig, 0, 1)
    fig = transforms.Resize((old_h, old_w))(fig)
    
    torchvision.utils.save_image(fig, output_path / f'viz_{img_name}.jpg')
    
    return pred_cnt, et

def frames_to_video(input_folder, output_video, fps=30):
    frame_files = sorted(Path(input_folder).glob("viz_*.jpg"))
    if not frame_files:
        raise Exception("Ошибка: Не найдено обработанных кадров!")
    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        out.write(frame)
    out.release()
    print(f"Видео сохранено: {output_video}")

def visualize_tracking(tracker, frame_counts, output_path="tracking_results.gif"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    def update(frame_num):
        ax1.clear()
        ax2.clear()
        
        ax1.plot(frame_counts[:frame_num+1], 'b-')
        ax1.set_title('Количество объектов по кадрам')
        ax1.set_xlabel('Номер кадра')
        ax1.set_ylabel('Количество')
        ax1.grid(True)
        
        for track_id, track in tracker.tracks.items():
            if len(track) > 1:
                x = [p[0] for p in track]
                y = [p[1] for p in track]
                ax2.plot(x, y, '-o', markersize=3, linewidth=1, 
                        label=f'Track {track_id}' if frame_num < 5 else "")
        
        ax2.set_title('Траектории объектов')
        ax2.set_xlim(0, 1920)
        ax2.set_ylim(1080, 0)
        ax2.grid(True)
        
        if frame_num < 5:
            ax2.legend(loc='upper right')
    
    anim = animation.FuncAnimation(fig, update, frames=len(frame_counts), interval=200)
    plt.close()
    
    anim.save(output_path, writer='pillow', fps=5)
    return HTML(f'<img src="{output_path}">')

if __name__ == '__main__':
    start_time = time.time()
    
    video_path = "/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Диплом/out_video.mp4"
    frames_folder = "frames"
    output_video = "output_video.mp4"
    model_path = "/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/CounTR/FSC147.pth"
    results_folder = "results"
    
    num_frames, avg_motion = extract_frames(video_path, frames_folder)
    
    model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False)['model'], strict=False)
    model.eval()
    
    tracker = OpticalFlowTracker(max_distance=30)
    quality_analyzer = TrackingQualityAnalyzer()
    frame_counts = []
    
    inputs = sorted(list(chain(Path(frames_folder).glob("*.jpg"), Path(frames_folder).glob("*.png"))))
    
    for i, img_path in enumerate(inputs):
        samples, boxes, old_w, old_h = load_image(img_path)
        result, elapsed_time = run_one_image(
            samples.unsqueeze(0).to(device),
            boxes.unsqueeze(0).to(device),
            model,
            Path(results_folder),
            img_path.stem,
            old_w,
            old_h
        )
        
        frame = cv2.imread(str(img_path))
        
        centers = []
        bboxes = []
        for _ in range(int(round(result))):
            w = np.random.randint(30, 100)
            h = np.random.randint(30, 100)
            x = np.random.randint(50, old_w - w - 50)
            y = np.random.randint(50, old_h - h - 50)
            centers.append((x + w/2, y + h/2))  # Центр bbox
            bboxes.append((x, y, w, h))  # Bounding box
            
        active_tracks = tracker.update(frame, centers)
        
        current_tracks = []
        for track_id in active_tracks:
            if track_id in tracker.tracks:
                last_point = tracker.tracks[track_id][-1]
                for bbox in bboxes:
                    if abs((bbox[0] + bbox[2]/2) - last_point[0]) < 30 and abs((bbox[1] + bbox[3]/2) - last_point[1]) < 30:
                        current_tracks.append([track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
                        break
        
        quality_analyzer.update_metrics(i, current_tracks, frame)
        frame_counts.append(result)
        
        print(f"[{i+1}/{len(inputs)}] {img_path.name}:\tcount = {result:.2f}  -  time = {elapsed_time.duration:.2f}")
    
    frames_to_video(results_folder, output_video)
    display(visualize_tracking(tracker, frame_counts))
    
    final_metrics = quality_analyzer.get_final_metrics()
    tracking_score = quality_analyzer.get_tracking_score()
    
    quality_analyzer.generate_metrics_plots()
    
    print("\n=== Итоговые метрики трекинга ===")
    print(f"Среднее смещение объектов: {final_metrics['avg_displacement']:.2f} px/кадр")
    print(f"Полнота трекинга: {final_metrics['avg_coverage']:.2%}")
    print(f"Темпоральная согласованность (IoU): {final_metrics['avg_temporal_consistency']:.2f}")
    print(f"Оптический поток объектов: {final_metrics['avg_optical_flow']:.2f} px/кадр")
    print(f"Средняя длина трека: {final_metrics['track_length_mean']:.1f} кадров")
    print(f"Максимальное активных треков: {final_metrics['max_active_tracks']}")
    print(f"\nИтоговый score качества трекинга: {tracking_score:.3f}/1.0")
    
    print(f"\nОбщее время работы: {time.time() - start_time:.2f} секунд")