"""
AST 기반 음원 분리 파이프라인 - 최종 통합 버전
템플릿 기반 분포 유사도를 활용한 정교한 음원 분리
"""

import numpy as np
import torch
import librosa
import soundfile as sf
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from scipy import signal
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.stats import wasserstein_distance
import json
import argparse
import os
import sys

# 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

# ==================== 상수 정의 ====================
class AudioConfig:
    """오디오 처리 설정"""
    SR = 16000  # 샘플링 레이트
    INPUT_SEC = 4.096  # 실제 입력 오디오 길이
    MODEL_SEC = 10.24  # AST 모델 처리 길이
    L_INPUT = int(round(INPUT_SEC * SR))  # 65,536 샘플
    L_MODEL = int(round(MODEL_SEC * SR))  # 163,840 샘플

class STFTConfig:
    """STFT 설정"""
    N_FFT = 400
    HOP = 160
    WINLEN = 400
    N_MELS = 128

class ASTConfig:
    """AST 모델 설정"""
    MAX_LENGTH = 1024
    FREQ_PATCHES = 12  # (128 - 16) / 10 + 1
    TIME_PATCHES = 101  # (1024 - 16) / 10 + 1

class SeparationConfig:
    """분리 파라미터"""
    SIMILARITY_THRESHOLD = 0.6  # 0.6 → 0.4로 낮춤 (더 많은 구간 분리)
    WEAK_MASK_FACTOR = 0.1  # 0.1 → 0.3으로 증가 (약한 구간도 더 강하게)
    MIN_ANCHOR_FRAMES = 8  # 최소 앵커 크기 (0.08초)
    MAX_ANCHOR_FRAMES = 128  # 최대 앵커 크기 (1.28초)
    CONTINUITY_GAP = 1  # 연속성 판단 최대 간격 (패치)
    ATTENTION_PERCENTILE = 80  # 상위 어텐션 퍼센타일
    MEDIUM_MASK_FACTOR = 0.6  # 중간 유사도 구간 마스크 강도

class WienerConfig:
    """Wiener 필터링 설정"""
    TANH_SCALE = 3.0  # tanh 함수 스케일링 (2.0 → 3.0)
    SIGMOID_SCALE = 10.0  # sigmoid 함수 스케일링 (8.0 → 10.0)
    SIGMOID_THRESHOLD = 0.6  # sigmoid 활성화 임계값 (0.7 → 0.6)
    ANCHOR_BOOST = 1.5  # 앵커 구간 부스트 (1.2 → 1.5)

# 편의를 위한 별칭
SR = AudioConfig.SR
INPUT_SEC = AudioConfig.INPUT_SEC
MODEL_SEC = AudioConfig.MODEL_SEC
L_INPUT = AudioConfig.L_INPUT
L_MODEL = AudioConfig.L_MODEL
N_FFT = STFTConfig.N_FFT
HOP = STFTConfig.HOP
WINLEN = STFTConfig.WINLEN
N_MELS = STFTConfig.N_MELS
AST_FREQ_PATCHES = ASTConfig.FREQ_PATCHES
AST_TIME_PATCHES = ASTConfig.TIME_PATCHES
SIMILARITY_THRESHOLD = SeparationConfig.SIMILARITY_THRESHOLD

# ==================== 데이터 클래스 ====================
@dataclass
class SeparationResult:
    """분리 결과 저장"""
    separated_audio: np.ndarray
    residual_audio: np.ndarray
    mask: np.ndarray
    anchor_frames: Tuple[int, int]
    energy_ratio: float
    classification: Dict[str, Any]
    attention_matrix: np.ndarray
    template_similarity: np.ndarray

# ==================== AST 모델 처리 ====================
class ASTProcessor:
    """AST 모델 처리 - CPU 전용"""
    
    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
        # CPU 강제 설정
        self.device = torch.device("cpu")
        torch.set_num_threads(4)
        print(f"디바이스: CPU (스레드: 4)")
        
        # 모델 로드
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        
        try:
            self.model = ASTForAudioClassification.from_pretrained(
                model_name,
                attn_implementation="eager",
                output_attentions=True,
                output_hidden_states=True
            )
        except:
            self.model = ASTForAudioClassification.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=True
            )
        
        # CPU 최적화
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("양자화 적용: 성공")
        except:
            print("양자화 적용: 실패 (일반 모드)")
        
        self.model.eval()
        print("AST 모델 로드 완료")
    
    def process(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """10.24초로 패딩하여 처리"""
        # 패딩
        padded = np.pad(audio, (0, max(0, L_MODEL - len(audio))), mode='constant')[:L_MODEL]
        
        # 특징 추출
        inputs = self.feature_extractor(padded, sampling_rate=SR, return_tensors="pt")
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 어텐션 추출
        if outputs.attentions:
            attention = outputs.attentions[-1].squeeze(0).mean(dim=0)
            attention_to_patches = attention[0, 1:]
        else:
            attention_to_patches = torch.ones(AST_FREQ_PATCHES * AST_TIME_PATCHES) / (AST_FREQ_PATCHES * AST_TIME_PATCHES)
        
        attention_matrix = attention_to_patches[:AST_FREQ_PATCHES * AST_TIME_PATCHES].reshape(
            AST_FREQ_PATCHES, AST_TIME_PATCHES
        ).cpu().numpy()
        
        # CLS/Distill 토큰 영향 제거: 좌상단 끝 부분 마스킹
        # 첫 번째 주파수 패치와 첫 번째 시간 패치의 어텐션을 낮춤
        attention_matrix[0, 0] *= 0.1  # 좌상단 끝 부분 강력히 억제
        if attention_matrix.shape[0] > 1:
            attention_matrix[1, 0] *= 0.3  # 두 번째 주파수 패치도 부분적으로 억제
        if attention_matrix.shape[1] > 1:
            attention_matrix[0, 1] *= 0.3  # 두 번째 시간 패치도 부분적으로 억제
        
        print(f"    🔍 CLS/distillation token masking applied to attention matrix")
        
        # 분류 결과
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        top5_probs, top5_indices = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)
        
        classification = {
            'predicted_class': self.model.config.id2label.get(top5_indices[0, 0].item(), "unknown"),
            'confidence': top5_probs[0, 0].item(),
            'top5_classes': [self.model.config.id2label.get(idx.item(), f"class_{idx}") 
                           for idx in top5_indices[0]],
            'top5_probs': top5_probs[0].cpu().numpy().tolist()
        }
        
        spectrogram = inputs['input_values'].squeeze(0).cpu().numpy()
        
        return attention_matrix, classification, spectrogram

# ==================== 오디오 처리 함수 ====================
def load_audio(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """오디오 로드 및 패딩"""
    audio, _ = librosa.load(file_path, sr=SR, mono=True)
    
    # 4.096초 버전
    audio_4sec = audio[:L_INPUT] if len(audio) > L_INPUT else np.pad(audio, (0, L_INPUT - len(audio)))
    
    # 10.24초 버전
    audio_10sec = audio[:L_MODEL] if len(audio) > L_MODEL else np.pad(audio, (0, L_MODEL - len(audio)))
    
    return audio_4sec, audio_10sec

def compute_stft(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT 계산"""
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN)
    mag = np.abs(stft)
    phase = np.angle(stft)
    return stft, mag, phase

# ==================== 템플릿 기반 분포 유사도 ====================
def calculate_distribution_similarity(template: np.ndarray, current: np.ndarray) -> float:
    """주파수-진폭 분포 유사도 계산 - 모든 값 양수 보장"""
    
    # 1. 음수 값 제거 및 최소값 보장 (양수 범위로 변환)
    template_positive = np.maximum(template, 1e-8)  # 음수 제거, 최소값 보장
    current_positive = np.maximum(current, 1e-8)    # 음수 제거, 최소값 보장
    
    # 2. 정규화 (확률 분포로 변환)
    template_norm = template_positive / (np.sum(template_positive) + 1e-8)
    current_norm = current_positive / (np.sum(current_positive) + 1e-8)
    
    # 3. 정규화된 값도 양수 보장
    template_norm = np.maximum(template_norm, 1e-8)
    current_norm = np.maximum(current_norm, 1e-8)
    
    # 4. 코사인 유사도
    cosine_sim = np.dot(template_norm, current_norm) / (
        np.linalg.norm(template_norm) * np.linalg.norm(current_norm) + 1e-8
    )
    cosine_sim = np.clip(cosine_sim, 0, 1)  # [0, 1] 범위 보장
    
    # 5. Earth Mover's Distance (양수 값만 사용)
    try:
        emd = wasserstein_distance(np.arange(len(template_norm)), 
                                   np.arange(len(current_norm)),
                                   template_norm, current_norm)
        emd_sim = np.exp(-emd / 10)
        emd_sim = np.clip(emd_sim, 0, 1)  # [0, 1] 범위 보장
    except Exception as e:
        print(f"    ⚠️ EMD calculation failed: {e}, using fallback")
        # EMD 실패 시 대안: L1 거리 기반 유사도
        l1_distance = np.sum(np.abs(template_norm - current_norm))
        emd_sim = np.exp(-l1_distance)
        emd_sim = np.clip(emd_sim, 0, 1)
    
    # 6. 스펙트럴 중심 비교
    freqs = np.arange(len(template))
    template_centroid = np.sum(freqs * template_norm)
    current_centroid = np.sum(freqs * current_norm)
    centroid_diff = abs(template_centroid - current_centroid) / len(template)
    centroid_sim = np.exp(-centroid_diff * 5)
    centroid_sim = np.clip(centroid_sim, 0, 1)  # [0, 1] 범위 보장
    
    # 7. 종합 유사도 (모든 값이 [0, 1] 범위)
    total_sim = 0.5 * cosine_sim + 0.3 * emd_sim + 0.2 * centroid_sim
    total_sim = np.clip(total_sim, 0, 1)  # 최종 [0, 1] 범위 보장
    
    return total_sim

# ==================== 헬퍼 함수들 ====================
def select_best_patch(stft_mag: np.ndarray, anchor_start: int, anchor_end: int, 
                     patch_size: int, num_patches: int) -> Tuple[np.ndarray, int, int]:
    """앵커 구간에서 가장 에너지가 높은 패치 선택"""
    patch_energies = []
    for i in range(num_patches):
        patch_start = anchor_start + i * patch_size
        patch_end = min(anchor_start + (i + 1) * patch_size, anchor_end)
        patch_spec = stft_mag[:, patch_start:patch_end]
        patch_energy = np.sum(patch_spec ** 2)
        patch_energies.append(patch_energy)
    
    # 가장 에너지가 높은 패치 선택
    best_patch_idx = np.argmax(patch_energies)
    best_patch_start = anchor_start + best_patch_idx * patch_size
    best_patch_end = min(anchor_start + (best_patch_idx + 1) * patch_size, anchor_end)
    best_patch_spec = stft_mag[:, best_patch_start:best_patch_end]
    
    print(f"    🔍 Selected best patch {best_patch_idx}/{num_patches} (energy: {patch_energies[best_patch_idx]:.3f})")
    
    return best_patch_spec, best_patch_start, best_patch_end

def calculate_patch_size(anchor_size: int) -> int:
    """앵커 크기에 따른 적응적 패치 크기 결정 - 최소 시간 구간으로 설정"""
    # 연속적인 시간 패턴을 확인할 수 있는 최소 시간 구간
    min_frames = 4  # 최소 4 프레임 (0.04초) - 매우 짧은 구간
    max_frames = 8  # 최대 8 프레임 (0.08초) - 짧은 구간
    
    if anchor_size <= min_frames:
        return anchor_size  # 전체 사용
    elif anchor_size <= max_frames:
        return min_frames  # 최소 구간 사용
    else:
        return min_frames  # 항상 최소 구간 사용 (시간 패턴 확인용)

def apply_smoothing_filters(signal: np.ndarray, is_2d: bool = False) -> np.ndarray:
    """신호에 다단계 스무딩 필터 적용"""
    if is_2d:
        # 2D 신호 (마스크)
        # 1단계: 주파수 축 방향 스무딩
        signal = gaussian_filter1d(signal, sigma=1.5, axis=0)
        # 2단계: 시간축 방향 스무딩
        signal = gaussian_filter1d(signal, sigma=1.0, axis=1)
        # 3단계: 2D 가우시안 필터
        from scipy.ndimage import gaussian_filter
        signal = gaussian_filter(signal, sigma=(0.8, 0.6))
    else:
        # 1D 신호 (유사도)
        from scipy.ndimage import median_filter
        signal = median_filter(signal, size=5)  # 중앙값 필터
        signal = uniform_filter1d(signal, size=3, mode='nearest')  # 평균 필터
    
    return np.clip(signal, 0, 1)

def validate_anchor_bounds(anchor_start: int, anchor_end: int, max_frames: int) -> Tuple[int, int]:
    """앵커 경계 검증 및 조정"""
    anchor_start = max(0, min(anchor_start, max_frames - 1))
    anchor_end = max(anchor_start + 1, min(anchor_end, max_frames))
    return anchor_start, anchor_end

# ==================== 앵커 선정 ====================
def find_attention_anchor(
    attention_matrix: np.ndarray,
    suppressed_regions: List[Tuple[int, int]],
    previous_peaks: List[Tuple[int, int]],
    audio_length: int
) -> Tuple[int, int, bool]:
    """완전히 유동적인 앵커 선정 - 어텐션 매트릭스에서 연속적인 부분만"""
    num_freq_patches, num_time_patches = attention_matrix.shape
    max_frames = audio_length // HOP
    valid_patches = int(INPUT_SEC / MODEL_SEC * num_time_patches)
    
    print(f"    🔍 Anchor selection: {num_freq_patches}x{num_time_patches} attention matrix")
    print(f"    🔍 Valid patches for 4.096s: {valid_patches}/{num_time_patches}")
    
    # 1. 억제된 영역 마스킹
    att_masked = attention_matrix.copy()
    for start, end in suppressed_regions:
        start_patch = int(start * num_time_patches / (L_MODEL // HOP))
        end_patch = int(end * num_time_patches / (L_MODEL // HOP))
        att_masked[:, max(0, start_patch):min(num_time_patches, end_patch)] *= 0.05
        print(f"    🔍 Suppressed region: patches {start_patch}-{end_patch}")
    
    # 2. 이전 패스의 상위 어텐션 영역 약화 (더 관대하게)
    if previous_peaks:
        for start, end in previous_peaks:
            # 0.3 → 0.5로 더 완화 (연속 구간을 찾을 수 있도록)
            att_masked[:, max(0, start):min(num_time_patches, end)] *= 0.5
            print(f"    🔍 Weakened previous peak: patches {start}-{end} (factor: 0.5)")
    
    # 3. 4.096초 범위 내에서만 탐색
    att_masked[:, valid_patches:] = 0
    
    # 4. 주파수 영역별 연속적인 활성화 구간 찾기
    print(f"    🔍 Finding continuous frequency-time attention regions...")
    
    # 각 주파수별로 높은 어텐션 영역 식별
    num_freq_patches = att_masked.shape[0]
    attention_threshold = np.percentile(att_masked, SeparationConfig.ATTENTION_PERCENTILE)
    
    print(f"    🔍 Attention threshold: {attention_threshold:.4f}")
    
    # 주파수별 활성화 마스크 생성
    freq_activation_maps = []
    for freq_idx in range(num_freq_patches):
        freq_attention = att_masked[freq_idx, :valid_patches]
        high_attention_mask = freq_attention >= attention_threshold
        freq_activation_maps.append(high_attention_mask)
    
    # 5. 주파수 영역별 연속 구간 찾기
    frequency_segments = []
    
    for freq_idx, activation_map in enumerate(freq_activation_maps):
        if not np.any(activation_map):
            continue
            
        # 해당 주파수에서 연속적인 시간 구간 찾기
        high_indices = np.where(activation_map)[0]
        
        # 연속 구간 그룹화
        continuous_segments = []
        current_segment = [high_indices[0]]
        
        for i in range(1, len(high_indices)):
            if high_indices[i] - high_indices[i-1] <= SeparationConfig.CONTINUITY_GAP:
                current_segment.append(high_indices[i])
            else:
                if len(current_segment) >= 2:  # 최소 2 패치 이상
                    continuous_segments.append((freq_idx, current_segment))
                current_segment = [high_indices[i]]
        
        # 마지막 구간 추가
        if len(current_segment) >= 2:
            continuous_segments.append((freq_idx, current_segment))
        
        frequency_segments.extend(continuous_segments)
    
    print(f"    🔍 Found {len(frequency_segments)} frequency-time segments")
    
    # 6. 주파수 영역별 연속성 점수 계산
    best_segment = None
    best_freq_idx = None
    best_score = -1
    
    for freq_idx, segment in frequency_segments:
        # 해당 주파수-시간 구간의 평균 어텐션
        segment_attention = np.mean([att_masked[freq_idx, i] for i in segment])
        
        # 점수 = 길이 × 평균 어텐션 × 주파수 가중치
        # 저주파(0-3)와 고주파(8-11)에 더 높은 가중치
        freq_weight = 1.0
        if freq_idx <= 3:  # 저주파
            freq_weight = 1.2
        elif freq_idx >= 8:  # 고주파
            freq_weight = 1.1
        
        score = len(segment) * segment_attention * freq_weight
        
        print(f"    🔍 Freq {freq_idx}, Time {segment[0]}-{segment[-1]}: length={len(segment)}, attention={segment_attention:.4f}, score={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_segment = segment
            best_freq_idx = freq_idx
    
    if best_segment is None:
        print(f"    ⚠️ No valid continuous segment found, using max attention fallback")
        # 폴백: 전체 어텐션 매트릭스에서 최대 어텐션 위치 찾기
        max_freq_idx, max_time_idx = np.unravel_index(np.argmax(att_masked), att_masked.shape)
        # 최대 어텐션 주변 3-5 패치를 앵커로 설정
        anchor_size = min(5, valid_patches // 8)  # 최대 5패치 또는 전체의 1/8
        start_patch = max(0, max_time_idx - anchor_size // 2)
        end_patch = min(valid_patches, start_patch + anchor_size)
        best_segment = list(range(start_patch, end_patch))
        best_freq_idx = max_freq_idx  # 폴백에서도 주파수 정보 보존
        print(f"    🔍 Fallback anchor: patches {start_patch}-{end_patch-1} (max attention at freq {max_freq_idx}, time {max_time_idx})")
    
    # 7. 앵커 구간을 연속적인 어텐션 길이에 맞춰 설정 (완전 유동적)
    anchor_start_patch = best_segment[0]
    anchor_end_patch = best_segment[-1] + 1  # +1로 패치 끝까지 포함
    
    # 패치를 프레임으로 변환
    anchor_start = int(anchor_start_patch * max_frames / valid_patches)
    anchor_end = int(anchor_end_patch * max_frames / valid_patches)
    
    # 경계 검증
    anchor_start, anchor_end = validate_anchor_bounds(anchor_start, anchor_end, max_frames)
    
    actual_anchor_size = anchor_end - anchor_start
    actual_anchor_time = actual_anchor_size * HOP / SR
    
    print(f"    🔍 Selected continuous segment: {len(best_segment)} patches")
    print(f"    🔍 Dynamic anchor: {actual_anchor_size} frames ({actual_anchor_time:.3f}s)")
    print(f"    🔍 Anchor range: {anchor_start}-{anchor_end} frames")
    print(f"    🔍 Time range: {anchor_start*HOP/SR:.3f}s - {anchor_end*HOP/SR:.3f}s")
    
    # 8. 크기 제한 적용
    min_frames = SeparationConfig.MIN_ANCHOR_FRAMES
    max_frames_limit = SeparationConfig.MAX_ANCHOR_FRAMES
    
    if actual_anchor_size < min_frames:
        center = (anchor_start + anchor_end) // 2
        anchor_start = max(0, center - min_frames // 2)
        anchor_end = min(max_frames, anchor_start + min_frames)
        print(f"    ⚠️ Expanded to minimum size: {anchor_end - anchor_start} frames")
    elif actual_anchor_size > max_frames_limit:
        center = (anchor_start + anchor_end) // 2
        anchor_start = max(0, center - max_frames_limit // 2)
        anchor_end = min(max_frames, anchor_start + max_frames_limit)
        print(f"    ⚠️ Limited to maximum size: {anchor_end - anchor_start} frames")
    
    return anchor_start, anchor_end, True, best_freq_idx

# ==================== 템플릿 기반 마스킹 ====================
def create_template_mask(
    stft_mag: np.ndarray,
    anchor_frames: Tuple[int, int],
    attention_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """템플릿 기반 분포 유사도 마스킹 - 수정 버전"""
    anchor_start, anchor_end = anchor_frames
    
    # 앵커 범위 검증
    anchor_start = min(anchor_start, stft_mag.shape[1] - 1)
    anchor_end = min(anchor_end, stft_mag.shape[1])
    
    if anchor_start >= anchor_end:
        return np.zeros_like(stft_mag), np.zeros(stft_mag.shape[1])
    
    # === 수정 1: 유동적 앵커 크기에 맞춘 템플릿 생성 ===
    anchor_spec = stft_mag[:, anchor_start:anchor_end]
    actual_anchor_size = anchor_end - anchor_start
    
    print(f"    🔍 Using dynamic anchor region: {actual_anchor_size} frames ({actual_anchor_size * HOP / SR:.3f}s)")
    
    # 앵커 크기에 따라 적응적 패치 크기 결정
    patch_size = calculate_patch_size(actual_anchor_size)
    
    if patch_size == actual_anchor_size:
        # 작은 앵커: 전체 사용
        best_patch_spec = anchor_spec
        best_patch_start = anchor_start
        best_patch_end = anchor_end
        print(f"    🔍 Small anchor: using entire region ({actual_anchor_size} frames)")
    else:
        # 중간/큰 앵커: 패치로 나누기
        num_patches = max(1, actual_anchor_size // patch_size)
        best_patch_spec, best_patch_start, best_patch_end = select_best_patch(
            stft_mag, anchor_start, anchor_end, patch_size, num_patches
        )
        print(f"    🔍 Anchor: selected best {patch_size}-frame patch from {num_patches} patches")
    
    # === 수정 2: 어텐션 매트릭스와 진폭 강조된 주파수 이용한 템플릿 생성 ===
    
    # 1. 기본 템플릿 (유동적 크기 패치의 평균)
    base_template = np.mean(best_patch_spec, axis=1)
    actual_patch_size = best_patch_spec.shape[1]
    print(f"    🔍 Template from {actual_patch_size}-frame patch ({actual_patch_size * HOP / SR:.3f}s)")
    
    # 2. 어텐션 매트릭스에서 해당 구간의 주파수 가중치 추출 (개선)
    attention_weights = np.ones(len(base_template))  # 기본값
    if attention_weights is not None and len(attention_weights.shape) > 1:
        # 앵커 구간에 해당하는 어텐션 패치들
        num_time_patches = attention_weights.shape[1]
        patch_start_attention = int(best_patch_start * num_time_patches / stft_mag.shape[1])
        patch_end_attention = int(best_patch_end * num_time_patches / stft_mag.shape[1])
        patch_start_attention = max(0, min(num_time_patches-1, patch_start_attention))
        patch_end_attention = max(0, min(num_time_patches, patch_end_attention))
        
        print(f"    🔍 Attention patch range: {patch_start_attention}-{patch_end_attention} (out of {num_time_patches})")
        
        # 해당 구간에서의 주파수별 어텐션 평균 (더 정확한 매핑)
        patch_attention_freq = np.mean(attention_weights[:, patch_start_attention:patch_end_attention], axis=1)
        
        # 어텐션 패치를 STFT 주파수 빈으로 매핑 (개선된 매핑)
        num_attention_freq_patches = attention_weights.shape[0]
        stft_freq_bins = len(base_template)
        
        print(f"    🔍 Mapping {num_attention_freq_patches} attention patches to {stft_freq_bins} STFT bins")
        
        for i in range(stft_freq_bins):
            # 더 정확한 매핑: STFT 주파수 빈을 어텐션 패치로 변환
            attention_patch_idx = int(i * num_attention_freq_patches / stft_freq_bins)
            attention_patch_idx = min(attention_patch_idx, num_attention_freq_patches - 1)
            attention_weights[i] = patch_attention_freq[attention_patch_idx]
        
        # 어텐션 가중치 정규화 및 강화
        attention_weights = np.maximum(attention_weights, 0.1)  # 최소값 보장
        attention_weights = attention_weights / (np.max(attention_weights) + 1e-8)  # 정규화
        
        print(f"    🔍 Attention weights range: {attention_weights.min():.3f} - {attention_weights.max():.3f}")
        print(f"    🔍 High attention frequencies: {np.sum(attention_weights > 0.7)}/{len(attention_weights)}")
    
    # 3. 진폭 강조된 주파수 식별
    amplitude_threshold = np.percentile(base_template, 75)  # 상위 25%
    amplitude_mask = base_template >= amplitude_threshold
    
    # 4. 어텐션과 진폭을 결합한 최종 템플릿 (개선)
    template = base_template.copy()
    
    # 어텐션 가중치 강화 적용 (더 강한 반영)
    attention_factor = 0.3 + 0.7 * attention_weights  # 0.3~1.0 범위
    template *= attention_factor
    
    # 진폭 강조된 주파수 추가 강화
    template[amplitude_mask] *= 1.8  # 1.5 → 1.8로 증가
    
    # 어텐션과 진폭이 모두 높은 주파수 특별 강화
    high_attention_mask = attention_weights > 0.7
    combined_mask = amplitude_mask & high_attention_mask
    template[combined_mask] *= 2.0  # 어텐션+진폭 모두 높은 주파수 2배 강화
    
    print(f"    🔍 Template enhancement: attention factor range {attention_factor.min():.3f}-{attention_factor.max():.3f}")
    print(f"    🔍 Amplitude-emphasized frequencies: {np.sum(amplitude_mask)}")
    print(f"    🔍 High attention frequencies: {np.sum(high_attention_mask)}")
    print(f"    🔍 Combined high attention+amplitude: {np.sum(combined_mask)}")
    
    # 템플릿 정규화 (양수 보장)
    template = np.maximum(template, 1e-8)
    template_norm = template / (np.sum(template) + 1e-8)
    
    print(f"    🔍 Template created: {len(template)} frequency bins, energy: {np.sum(template**2):.3f}")
    print(f"    🔍 Template time span: {actual_patch_size} frames ({actual_patch_size * HOP / SR:.3f}s)")
    print(f"    🔍 Attention-weighted template with {np.sum(amplitude_mask)}/{len(template)} amplitude-emphasized frequencies")
    
    # === 수정 3: 유동적 크기 템플릿 기반 유사도 계산 ===
    num_frames = stft_mag.shape[1]
    similarity = np.zeros(num_frames)
    
    print(f"    🔍 Computing similarity for {num_frames} frames using {actual_patch_size}-frame template")
    print(f"    🔍 Template represents minimal time pattern for continuous comparison")
    
    for t in range(num_frames):
        current = stft_mag[:, t]
        
        # 현재 프레임 정규화 (양수 보장)
        current = np.maximum(current, 1e-8)
        current_norm = current / (np.sum(current) + 1e-8)
        
        # 1. 코사인 유사도 (가장 안정적)
        cosine_sim = np.dot(template_norm, current_norm) / (
            np.linalg.norm(template_norm) * np.linalg.norm(current_norm) + 1e-8
        )
        cosine_sim = np.clip(cosine_sim, 0, 1)
        
        # 2. 에너지 비율 유사도
        template_energy = np.sum(template ** 2)
        current_energy = np.sum(current ** 2)
        if template_energy > 1e-8 and current_energy > 1e-8:
            energy_ratio = min(template_energy, current_energy) / max(template_energy, current_energy)
        else:
            energy_ratio = 0
        
        # 3. 스펙트럴 중심 유사도
        freqs = np.arange(len(template))
        template_centroid = np.sum(freqs * template_norm)
        current_centroid = np.sum(freqs * current_norm)
        centroid_diff = abs(template_centroid - current_centroid) / len(template)
        centroid_sim = np.exp(-centroid_diff * 3)  # 더 강한 가중치
        
        # 4. 주파수 분포 유사도 (추가)
        # 템플릿과 현재 프레임의 주파수 분포 비교
        freq_dist_sim = np.corrcoef(template_norm, current_norm)[0, 1]
        if np.isnan(freq_dist_sim):
            freq_dist_sim = 0
        freq_dist_sim = np.clip(freq_dist_sim, 0, 1)
        
        # 종합 유사도 (가중치 조정 - 주파수 분포 추가)
        similarity[t] = 0.4 * cosine_sim + 0.2 * energy_ratio + 0.2 * centroid_sim + 0.2 * freq_dist_sim
        similarity[t] = np.clip(similarity[t], 0, 1)
    
    # === 수정 3: 앵커 구간도 실제 유사도 계산 (강제 보정 제거) ===
    # 앵커 구간도 실제 유사도로 계산하여 자연스러운 비교 수행
    # similarity[anchor_start:anchor_end] = 1.0  # 이 부분 제거
    
    # === 수정 4: 시간적 스무딩 (급격한 변화 방지) ===
    similarity = apply_smoothing_filters(similarity, is_2d=False)
    
    # 앵커 구간 재보정 제거 - 실제 유사도 유지
    # similarity[anchor_start:anchor_end] = np.maximum(
    #     similarity[anchor_start:anchor_end], 0.95
    # )
    
    # === 수정 5: 2D 마스크 생성 ===
    mask = np.zeros_like(stft_mag)
    
    for t in range(num_frames):
        if similarity[t] >= SeparationConfig.SIMILARITY_THRESHOLD:
            # 높은 유사도: 매우 강하게 분리
            mask[:, t] = similarity[t] * 1.8  # 80% 추가 강화
        elif similarity[t] >= SeparationConfig.SIMILARITY_THRESHOLD * 0.7:
            # 중간-높은 유사도: 강하게 분리
            mask[:, t] = similarity[t] * 1.2  # 20% 추가 강화
        elif similarity[t] >= SeparationConfig.SIMILARITY_THRESHOLD * 0.4:
            # 중간 유사도: 보통 강도
            mask[:, t] = similarity[t] * 0.6
        elif similarity[t] >= SeparationConfig.SIMILARITY_THRESHOLD * 0.2:
            # 낮은 유사도: 아주 약하게 분리
            mask[:, t] = similarity[t] * 0.15
    else:
            # 매우 낮은 유사도: 거의 분리하지 않음
            mask[:, t] = similarity[t] * 0.05
    
    # === 수정 6: 주파수별 가중치 적용 및 원본 진폭 제한 ===
    if anchor_spec.size > 0:
        # 앵커에서 중요한 주파수 식별 (더 강한 가중치)
        freq_importance = np.percentile(anchor_spec, 80, axis=1)  # 80 percentile로 상향
        freq_importance = freq_importance / (np.max(freq_importance) + 1e-8)
        
        # 중요 주파수 강조 (어텐션 기반 가중치 적용)
        for f in range(mask.shape[0]):
            # 기본 중요도 가중치
            importance_factor = 0.5 + 1.0 * freq_importance[f]
            
            # 어텐션 가중치 추가 (해당 주파수가 어텐션에서 높은 경우)
            if f < len(attention_weights):
                attention_factor = 0.8 + 0.4 * attention_weights[f]  # 0.8~1.2 범위
            else:
                attention_factor = 1.0
            
            # 최종 가중치 = 중요도 × 어텐션
            weight_factor = importance_factor * attention_factor
            mask[f, :] *= weight_factor
        
        # 원본 오디오 진폭 제한: 마스크가 원본을 넘지 않도록
        # 각 시간 프레임별로 원본 스펙트로그램 대비 마스크 강도 제한
        for t in range(mask.shape[1]):
            original_energy = np.sum(stft_mag[:, t] ** 2)
            if original_energy > 1e-8:
                # 마스크 적용 후 에너지가 원본을 넘지 않도록 제한 (100%)
                max_energy_ratio = 1.0
                current_masked_energy = np.sum((stft_mag[:, t] * mask[:, t]) ** 2)
                
                if current_masked_energy > original_energy * max_energy_ratio:
                    # 스케일링 팩터 계산
                    scale_factor = np.sqrt(original_energy * max_energy_ratio / current_masked_energy)
                    mask[:, t] *= scale_factor
                    
                    print(f"    🔧 Frame {t}: Energy limited to {max_energy_ratio*100}% of original")
        
        print(f"    📊 Frequency weighting applied: {np.sum(freq_importance > 0.5)}/{len(freq_importance)} important bins")
    
    # 최종 정규화 및 스무딩 (주파수와 시간 모두)
    mask = apply_smoothing_filters(mask, is_2d=True)
    
    print(f"    🔍 Enhanced smoothing applied: frequency axis (σ=1.5), time axis (σ=1.0), 2D (σ=0.8,0.6)")
    
    return mask, similarity

# ==================== Wiener 필터링 ====================
def apply_adaptive_wiener(
    stft: np.ndarray,
    mask: np.ndarray,
    anchor_frames: Tuple[int, int]
) -> np.ndarray:
    """적응적 Wiener 필터링 - tanh/sigmoid 기반 강도 조절"""
    anchor_start, anchor_end = anchor_frames
    
    # 노이즈 추정
    noise_mask = 1.0 - mask
    noise_power = np.abs(stft) ** 2 * noise_mask
    noise_estimate = np.mean(noise_power, axis=1, keepdims=True) + 1e-8
    
    # 신호 파워
    signal_power = np.abs(stft) ** 2 * mask
    
    # 기본 Wiener gain 계산
    wiener_gain = signal_power / (signal_power + noise_estimate)
    wiener_gain = np.clip(wiener_gain, 0, 1)
    
    # tanh 기반 강도 조절: 높은 부분은 강하게, 약한 부분은 약하게
    # mask 값에 따라 tanh 함수로 강도 조절
    # mask가 높을수록 (1에 가까울수록) 더 강한 필터링 적용
    
    # 1. tanh 기반 강도 조절 (설정 가능한 스케일링)
    intensity_factor = 0.5 + 0.5 * np.tanh(WienerConfig.TANH_SCALE * (mask - 0.5))
    
    # 2. sigmoid 기반 추가 강화 (설정 가능한 임계값과 스케일)
    sigmoid_factor = 1 / (1 + np.exp(-WienerConfig.SIGMOID_SCALE * (mask - WienerConfig.SIGMOID_THRESHOLD)))
    
    # 3. 최종 강도 조절 (tanh + sigmoid 결합)
    final_intensity = intensity_factor * (0.7 + 0.3 * sigmoid_factor)
    
    # 4. Wiener gain에 강도 적용
    enhanced_wiener_gain = wiener_gain * final_intensity
    
    # 5. 앵커 구간 특별 처리 (설정 가능한 부스트)
    enhanced_wiener_gain[:, anchor_start:anchor_end] *= WienerConfig.ANCHOR_BOOST
    
    # 6. 최종 클리핑
    enhanced_wiener_gain = np.clip(enhanced_wiener_gain, 0, 1)
    
    print(f"    🔍 Wiener filtering: tanh/sigmoid intensity applied")
    print(f"    🔍 Intensity range: {final_intensity.min():.3f} - {final_intensity.max():.3f}")
    print(f"    🔍 Anchor boost: {WienerConfig.ANCHOR_BOOST}x")
    
    return stft * enhanced_wiener_gain

# ==================== 단일 패스 처리 ====================
def process_single_pass(
    audio_4sec: np.ndarray,
    audio_10sec: np.ndarray,
    ast_processor: ASTProcessor,
    suppressed_regions: List[Tuple[int, int]],
    previous_peaks: List[Tuple[int, int]],
    pass_idx: int
) -> Optional[SeparationResult]:
    """단일 패스 음원 분리"""
    
    print(f"  처리 중...")
    
    # AST 처리 (10.24초)
    attention_matrix, classification, ast_spec = ast_processor.process(audio_10sec)
    
    # STFT (4.096초)
    stft, mag, phase = compute_stft(audio_4sec)
    
    # 앵커 선정
    anchor_start, anchor_end, is_valid, anchor_freq_idx = find_attention_anchor(
        attention_matrix, suppressed_regions, previous_peaks, len(audio_4sec)
    )
    
    if not is_valid:
        return None
    
    anchor_time = (anchor_start * HOP / SR, anchor_end * HOP / SR)
    print(f"  앵커: {anchor_time[0]:.3f}s - {anchor_time[1]:.3f}s")
    
    # 템플릿 마스킹
    mask, similarity = create_template_mask(mag, (anchor_start, anchor_end), attention_matrix)
    
    # Wiener 필터링
    filtered_stft = apply_adaptive_wiener(stft, mask, (anchor_start, anchor_end))
    
    # 역변환
    separated = librosa.istft(filtered_stft, hop_length=HOP, win_length=WINLEN)
    
    # 길이 조정
    if len(separated) < L_INPUT:
        separated = np.pad(separated, (0, L_INPUT - len(separated)))
    else:
        separated = separated[:L_INPUT]
    
    # 잔여 계산
    residual = audio_4sec - separated
    
    # 에너지 비율
    energy_ratio = np.sum(separated ** 2) / (np.sum(audio_4sec ** 2) + 1e-8)
    
    return SeparationResult(
        separated_audio=separated,
        residual_audio=residual,
        mask=mask,
        anchor_frames=(anchor_start, anchor_end),
        energy_ratio=energy_ratio,
        classification=classification,
        attention_matrix=attention_matrix,
        template_similarity=similarity
    )

# ==================== 멀티패스 분리 ====================
def multi_pass_separation(
    audio_4sec: np.ndarray,
    audio_10sec: np.ndarray,
    ast_processor: ASTProcessor,
    max_passes: int = 2
) -> List[SeparationResult]:
    """멀티패스 음원 분리"""
    results = []
    suppressed_regions = []
    previous_peaks = []
    
    current_4sec = audio_4sec.copy()
    current_10sec = audio_10sec.copy()
    
    for i in range(max_passes):
        print(f"\n=== Pass {i + 1}/{max_passes} ===")
        
        # 디버그: 현재 사용할 오디오 정보 출력
        if i == 0:
            print(f"  🔍 Using original audio: {len(current_4sec)} samples")
        else:
            print(f"  🔍 Using residual audio: {len(current_4sec)} samples")
            print(f"  🔍 Residual energy: {np.sum(current_4sec**2):.2f}")
        
        result = process_single_pass(
            current_4sec, current_10sec, ast_processor,
            suppressed_regions, previous_peaks, i
        )
        
        if result is None or result.energy_ratio < 0.0001:  # 임계값을 더욱 낮춤 (0.0005 → 0.0001)
            print(f"  종료: 에너지 부족 (ratio: {result.energy_ratio if result else 0:.6f})")
            break
        
        # 상위 어텐션 영역 저장
        att_mean = np.mean(result.attention_matrix, axis=0)
        threshold = np.percentile(att_mean, SeparationConfig.ATTENTION_PERCENTILE)
        peaks = np.where(att_mean > threshold)[0]
        if len(peaks) > 0:
            previous_peaks.append((peaks[0], peaks[-1]))
        
        results.append(result)
        suppressed_regions.append(result.anchor_frames)
        
        # 잔여 오디오로 업데이트
        current_4sec = result.residual_audio
        current_10sec = np.pad(result.residual_audio, (0, max(0, L_MODEL - len(result.residual_audio))))[:L_MODEL]
        
        print(f"  클래스: {result.classification['predicted_class']}")
        print(f"  신뢰도: {result.classification['confidence']:.3f}")
        print(f"  에너지: {result.energy_ratio:.3f}")
    
    return results

# ==================== 시각화 ====================
def create_debug_plot(
    original_audio: np.ndarray,
    results: List[SeparationResult],
    output_dir: Path
):
    """디버그 시각화"""
    for idx, result in enumerate(results):
        # 각 패스별로 올바른 입력 오디오 선택
        if idx == 0:
            # Pass 1: 원본 오디오
            input_audio = original_audio
        else:
            # Pass 2+: 이전 패스의 잔여 오디오
            input_audio = results[idx-1].residual_audio
        fig = plt.figure(figsize=(16, 10))
        
        # 레이아웃 설정
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 시간축
        time_axis = np.arange(len(input_audio)) / SR
        
        # 1. 입력 오디오 (Pass 1: 원본, Pass 2+: 잔여)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_axis, input_audio, 'b-', alpha=0.7)
        if idx == 0:
            ax1.set_title('Original Audio')
        else:
            ax1.set_title(f'Residual Audio from Pass {idx}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_ylim(-1, 1)  # -1에서 1 범위로 통일
        ax1.grid(True, alpha=0.3)
        
        # 2. 분리된 오디오
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_axis, result.separated_audio, 'g-', alpha=0.7)
        ax2.set_title(f'Separated: {result.classification["predicted_class"]}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_ylim(-1, 1)  # -1에서 1 범위로 통일
        ax2.grid(True, alpha=0.3)
        
        # 3. 잔여 오디오
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(time_axis, result.residual_audio, 'r-', alpha=0.7)
        ax3.set_title('Residual Audio')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.set_ylim(-1, 1)  # -1에서 1 범위로 통일
        ax3.grid(True, alpha=0.3)
        
        # 4. 템플릿 유사도
        ax4 = fig.add_subplot(gs[1, 0])
        time_frames = np.arange(len(result.template_similarity)) * HOP / SR
        ax4.plot(time_frames, result.template_similarity, 'purple', linewidth=2)
        # 5단계 유사도 임계값 표시
        ax4.axhline(y=SeparationConfig.SIMILARITY_THRESHOLD, color='red', linestyle='-', linewidth=2,
                   label=f'High={SeparationConfig.SIMILARITY_THRESHOLD}')
        ax4.axhline(y=SeparationConfig.SIMILARITY_THRESHOLD * 0.7, color='orange', linestyle='--', 
                   label=f'Med-High={SeparationConfig.SIMILARITY_THRESHOLD * 0.7:.2f}')
        ax4.axhline(y=SeparationConfig.SIMILARITY_THRESHOLD * 0.4, color='yellow', linestyle=':', 
                   label=f'Medium={SeparationConfig.SIMILARITY_THRESHOLD * 0.4:.2f}')
        ax4.axhline(y=SeparationConfig.SIMILARITY_THRESHOLD * 0.2, color='lightblue', linestyle=':', 
                   label=f'Weak={SeparationConfig.SIMILARITY_THRESHOLD * 0.2:.2f}')
        
        # 5단계 색상 구분
        ax4.fill_between(time_frames, 0, result.template_similarity, 
                         where=(result.template_similarity >= SeparationConfig.SIMILARITY_THRESHOLD),
                         color='red', alpha=0.4, label='Very High (1.8x)')
        ax4.fill_between(time_frames, 0, result.template_similarity, 
                         where=((result.template_similarity >= SeparationConfig.SIMILARITY_THRESHOLD * 0.7) & 
                                (result.template_similarity < SeparationConfig.SIMILARITY_THRESHOLD)),
                         color='orange', alpha=0.3, label='High (1.2x)')
        ax4.fill_between(time_frames, 0, result.template_similarity, 
                         where=((result.template_similarity >= SeparationConfig.SIMILARITY_THRESHOLD * 0.4) & 
                                (result.template_similarity < SeparationConfig.SIMILARITY_THRESHOLD * 0.7)),
                         color='yellow', alpha=0.2, label='Medium (0.6x)')
        ax4.fill_between(time_frames, 0, result.template_similarity, 
                         where=((result.template_similarity >= SeparationConfig.SIMILARITY_THRESHOLD * 0.2) & 
                                (result.template_similarity < SeparationConfig.SIMILARITY_THRESHOLD * 0.4)),
                         color='lightblue', alpha=0.1, label='Weak (0.15x)')
        ax4.fill_between(time_frames, 0, result.template_similarity, 
                         where=(result.template_similarity < SeparationConfig.SIMILARITY_THRESHOLD * 0.2),
                         color='lightgray', alpha=0.05, label='Very Weak (0.05x)')
        ax4.set_title('Template Similarity')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Similarity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 분리 마스크
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(result.mask, aspect='auto', origin='lower', cmap='hot', interpolation='bilinear')
        ax5.set_title('Separation Mask')
        ax5.set_xlabel('Time frames')
        ax5.set_ylabel('Frequency bins')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        # 앵커 영역 표시
        anchor_start, anchor_end = result.anchor_frames
        ax5.axvline(x=anchor_start, color='cyan', linestyle='--', linewidth=2)
        ax5.axvline(x=anchor_end, color='cyan', linestyle='--', linewidth=2)
        
        # 6. STFT 스펙트로그램
        ax6 = fig.add_subplot(gs[1, 2])
        _, mag, _ = compute_stft(input_audio)
        db_spec = librosa.amplitude_to_db(mag, ref=np.max)
        im6 = ax6.imshow(db_spec, aspect='auto', origin='lower', cmap='magma', interpolation='bilinear')
        ax6.set_title('STFT Spectrogram (dB)')
        ax6.set_xlabel('Time frames')
        ax6.set_ylabel('Frequency bins')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
        
        # 7. 어텐션 매트릭스 (차원 일치)
        ax7 = fig.add_subplot(gs[2, 0])
        # 어텐션 매트릭스는 [freq_patches, time_patches] 형태
        # imshow는 기본적으로 origin='upper'이므로 주파수가 위에서 아래로
        # 다른 시각화와 일치시키기 위해 origin='lower'로 설정
        im7 = ax7.imshow(result.attention_matrix, aspect='auto', origin='lower', cmap='coolwarm', interpolation='bilinear')
        ax7.set_title('AST Attention Matrix (Frequency: 0→High)')
        ax7.set_xlabel('Time patches')
        ax7.set_ylabel('Frequency patches (0→High)')
        
        # 4.096초 경계 표시
        valid_patches = int(INPUT_SEC / MODEL_SEC * AST_TIME_PATCHES)
        ax7.axvline(x=valid_patches, color='green', linestyle='--', linewidth=2, label='4.096s')
        
        # 앵커 영역 표시 (4.096초 시간대에 맞춰)
        anchor_start, anchor_end = result.anchor_frames
        # STFT 프레임을 어텐션 패치로 변환 (4.096초 기준)
        # 4.096초에 해당하는 유효 패치 수 계산
        valid_patches = int(INPUT_SEC / MODEL_SEC * AST_TIME_PATCHES)
        anchor_start_patch = int(anchor_start * valid_patches / (L_INPUT // HOP))
        anchor_end_patch = int(anchor_end * valid_patches / (L_INPUT // HOP))
        anchor_start_patch = max(0, min(valid_patches-1, anchor_start_patch))
        anchor_end_patch = max(0, min(valid_patches, anchor_end_patch))
        
        # 앵커 영역 빨간색 박스로 표시
        from matplotlib.patches import Rectangle
        rect = Rectangle((anchor_start_patch, 0), anchor_end_patch - anchor_start_patch, 
                        result.attention_matrix.shape[0], linewidth=3, 
                        edgecolor='red', facecolor='none', alpha=0.8)
        ax7.add_patch(rect)
        
        # 앵커 텍스트 표시
        ax7.text(anchor_start_patch + (anchor_end_patch - anchor_start_patch) / 2, 
                result.attention_matrix.shape[0] * 0.9,
                f'', 
                ha='center', va='top', fontsize=10, color='red', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax7.legend()
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        # 8. 에너지 분포
        ax8 = fig.add_subplot(gs[2, 1])
        energies = {
            'Input': np.sum(input_audio ** 2),
            'Separated': np.sum(result.separated_audio ** 2),
            'Residual': np.sum(result.residual_audio ** 2)
        }
        bars = ax8.bar(energies.keys(), energies.values(), color=['blue', 'green', 'red'])
        ax8.set_title('Energy Distribution')
        ax8.set_ylabel('Energy')
        
        # 에너지 비율 표시
        for bar, (name, value) in zip(bars, energies.items()):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value/energies["Input"]*100:.1f}%',
                    ha='center', va='bottom')
        
        # 9. 분류 결과
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # 분류 정보 텍스트
        info_text = f"Classification Results\n" + "="*30 + "\n"
        info_text += f"Predicted: {result.classification['predicted_class']}\n"
        info_text += f"Confidence: {result.classification['confidence']:.2%}\n\n"
        info_text += "Top 5 Classes:\n"
        for cls, prob in zip(result.classification['top5_classes'][:5], 
                            result.classification['top5_probs'][:5]):
            info_text += f"  • {cls}: {prob:.2%}\n"
        info_text += f"\nEnergy Ratio: {result.energy_ratio:.3f}\n"
        info_text += f"Anchor: {anchor_start*HOP/SR:.3f}s - {anchor_end*HOP/SR:.3f}s"
        
        ax9.text(0.1, 0.9, info_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 전체 제목
        fig.suptitle(f'Pass {idx + 1} - Audio Source Separation Analysis', fontsize=14, fontweight='bold')
        
        # 저장
        output_path = output_dir / f'debug_pass_{idx+1}.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  시각화 저장: {output_path}")

# ==================== 메인 파이프라인 ====================
class SourceSeparationPipeline:
    """AST 기반 음원 분리 파이프라인"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.ast_processor = ASTProcessor()
        
    def process(self, audio_path: str, visualize: bool = True) -> Dict[str, Any]:
        """오디오 파일 처리"""
        print(f"\n{'='*60}")
        print(f"처리 시작: {audio_path}")
        print(f"{'='*60}")
        
        # 오디오 로드
        audio_4sec, audio_10sec = load_audio(audio_path)
        print(f"오디오 로드: {INPUT_SEC}초 → {MODEL_SEC}초 패딩")
        
        # 멀티패스 분리
        results = multi_pass_separation(audio_4sec, audio_10sec, self.ast_processor)
        
        # 결과 저장
        output_data = {
            'input_file': str(audio_path),
            'input_duration': INPUT_SEC,
            'model_duration': MODEL_SEC,
            'num_sources': len(results),
            'sources': []
        }
        
        print(f"\n{'='*60}")
        print(f"결과 저장")
        print(f"{'='*60}")
        
        for idx, result in enumerate(results, 1):
            # 오디오 저장
            audio_path = self.output_dir / f'separated_{idx}.wav'
            sf.write(audio_path, result.separated_audio, SR)
            
            # 메타데이터
            anchor_time_start = result.anchor_frames[0] * HOP / SR
            anchor_time_end = result.anchor_frames[1] * HOP / SR
            
            source_info = {
                'index': idx,
                'audio_file': str(audio_path),
                'class': result.classification['predicted_class'],
                'confidence': float(result.classification['confidence']),
                'top5_classes': result.classification['top5_classes'],
                'top5_probs': result.classification['top5_probs'],
                'energy_ratio': float(result.energy_ratio),
                'anchor_time': {
                    'start': float(anchor_time_start),
                    'end': float(anchor_time_end),
                    'duration': float(anchor_time_end - anchor_time_start)
                }
            }
            output_data['sources'].append(source_info)
            
            print(f"소스 {idx}:")
            print(f"  - 클래스: {source_info['class']}")
            print(f"  - 신뢰도: {source_info['confidence']:.2%}")
            print(f"  - 에너지: {source_info['energy_ratio']:.2%}")
            print(f"  - 파일: {audio_path}")
        
        # 잔여 오디오 저장
        if results:
            residual_path = self.output_dir / 'residual.wav'
            sf.write(residual_path, results[-1].residual_audio, SR)
            output_data['residual_file'] = str(residual_path)
            print(f"잔여: {residual_path}")
        
        # 메타데이터 저장
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"메타데이터: {metadata_path}")
        
        # 시각화
        if visualize:
            print(f"\n시각화 생성 중...")
            create_debug_plot(audio_4sec, results, self.output_dir)
        
        print(f"\n{'='*60}")
        print(f"처리 완료!")
        print(f"{'='*60}")
        
        return output_data

# ==================== 유틸리티 함수 ====================
def analyze_audio(audio: np.ndarray, sr: int = SR) -> Dict[str, Any]:
    """오디오 분석"""
    # 기본 통계
    stats = {
        'duration': len(audio) / sr,
        'samples': len(audio),
        'mean': float(np.mean(audio)),
        'std': float(np.std(audio)),
        'max': float(np.max(np.abs(audio))),
        'rms': float(np.sqrt(np.mean(audio ** 2))),
        'zero_crossings': int(np.sum(np.diff(np.sign(audio)) != 0) / 2)
    }
    
    # 스펙트럴 특성
    _, mag, _ = compute_stft(audio)
    spectral_centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=mag, sr=sr)[0]
    
    stats['spectral'] = {
        'centroid_mean': float(np.mean(spectral_centroid)),
        'centroid_std': float(np.std(spectral_centroid)),
        'rolloff_mean': float(np.mean(spectral_rolloff)),
        'rolloff_std': float(np.std(spectral_rolloff))
    }
    
    return stats

def evaluate_separation(original: np.ndarray, separated: List[np.ndarray]) -> Dict[str, float]:
    """분리 품질 평가"""
    original_energy = np.sum(original ** 2)
    separated_energy = sum(np.sum(s ** 2) for s in separated)
    
    # 재구성
    reconstructed = np.sum(separated, axis=0) if len(separated) > 0 else np.zeros_like(original)
    mse = np.mean((original - reconstructed) ** 2)
    
    # SDR (Signal-to-Distortion Ratio)
    sdr = 10 * np.log10(original_energy / (mse + 1e-8)) if mse > 0 else float('inf')
    
    return {
        'energy_preservation': float(separated_energy / (original_energy + 1e-8)),
        'reconstruction_mse': float(mse),
        'sdr_db': float(sdr),
        'num_sources': len(separated)
    }

# ==================== 메인 실행 ====================
def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='AST 기반 음원 분리 파이프라인',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python separator.py --input audio.wav
  python separator.py --input audio.wav --output results --no-viz
  python separator.py --input audio.wav --analyze
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='입력 오디오 파일 경로')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='출력 디렉토리 (기본: output)')
    parser.add_argument('--no-viz', action='store_true',
                       help='시각화 비활성화')
    parser.add_argument('--analyze', action='store_true',
                       help='오디오 분석 정보 출력')
    
    args = parser.parse_args()
    
    # 입력 확인
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    
    # 헤더 출력
    print("\n" + "="*60)
    print("AST 기반 음원 분리 파이프라인 v1.0")
    print("="*60)
    print(f"설정:")
    print(f"  - 입력: {INPUT_SEC}초")
    print(f"  - 모델: {MODEL_SEC}초")
    print(f"  - 디바이스: CPU")
    print(f"  - 샘플링: {SR} Hz")
    print(f"  - STFT: N={N_FFT}, Hop={HOP}")
    print("="*60)
    
    try:
        # 파이프라인 실행
        pipeline = SourceSeparationPipeline(output_dir=args.output)
        
        # 분석 모드
        if args.analyze:
            print("\n오디오 분석 중...")
            audio, _ = load_audio(str(input_path))
            analysis = analyze_audio(audio)
            print("\n오디오 특성:")
            print(f"  - 길이: {analysis['duration']:.3f}초")
            print(f"  - RMS: {analysis['rms']:.4f}")
            print(f"  - 영교차율: {analysis['zero_crossings']}")
            print(f"  - 스펙트럴 중심: {analysis['spectral']['centroid_mean']:.1f} Hz")
            print(f"  - 스펙트럴 롤오프: {analysis['spectral']['rolloff_mean']:.1f} Hz")
        
        # 처리
        result = pipeline.process(str(input_path), visualize=not args.no_viz)
        
        # 품질 평가
        if result['num_sources'] > 0:
            print("\n분리 품질 평가:")
            audio, _ = load_audio(str(input_path))
            separated = []
            for source in result['sources']:
                sep_audio, _ = librosa.load(source['audio_file'], sr=SR)
                separated.append(sep_audio[:len(audio)])
            
            quality = evaluate_separation(audio, separated)
            print(f"  - 에너지 보존: {quality['energy_preservation']:.2%}")
            print(f"  - SDR: {quality['sdr_db']:.1f} dB")
            print(f"  - MSE: {quality['reconstruction_mse']:.6f}")
        
        # 완료
        print(f"\n결과 저장 위치: {Path(args.output).absolute()}")
        print("\n처리가 성공적으로 완료되었습니다!")
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
