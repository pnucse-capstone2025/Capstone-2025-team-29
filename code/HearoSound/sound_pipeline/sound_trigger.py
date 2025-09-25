#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound Trigger Module
- 100dB 이상 소리 감지 시 1.024x4초 녹음
- triggered_record.py 기반으로 구현
"""

import pyaudio
import wave
import numpy as np
import time
import sys
import os
import threading
from typing import Optional, Tuple

# ===== 설정 =====
FORMAT = pyaudio.paInt16
RATE = 16000
CHUNK = 1024
THRESHOLD_DB = 30  # 30dB 임계값 (테스트용)
RECORD_SECONDS = 1.024 * 4  # 1.024 x 4초 = 4.096초
TARGET_DEVICE_KEYWORDS = ("ReSpeaker", "seeed", "SEEED")  # 장치명에 포함될 키워드

class SoundTrigger:
    def __init__(self, output_dir: str = "recordings", led_controller=None):
        """
        Sound Trigger 초기화
        
        Args:
            output_dir: 녹음 파일 저장 디렉토리
            led_controller: LED 컨트롤러 인스턴스 (wake up용)
        """
        self.output_dir = output_dir
        self.led_controller = led_controller
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.device_index = None
        self.max_in_ch = 0
        self.desired_channels = 0
        
        # 동시 녹음 제한 (최대 2개)
        self.active_recordings = 0
        self.max_concurrent_recordings = 2
        self.recording_lock = threading.Lock()
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 장치 초기화
        self._initialize_device()
        
    def _initialize_device(self):
        """ReSpeaker 장치 초기화"""
        self.device_index, self.max_in_ch = self._find_respeaker_input_device()
        
        # ReSpeaker v2.0이면 6채널(0~3: 마이크, 4: reference, 5: post-processed)을 시도
        # 안 되면 사용 가능한 최대 입력 채널로 열고, 최종적으로 1채널 변환 저장
        self.desired_channels = 6 if self.max_in_ch >= 6 else self.max_in_ch if self.max_in_ch > 0 else 1
        
        
        info = self.p.get_device_info_by_index(self.device_index)
        print(f"[Device] index={self.device_index}, name='{info.get('name')}', maxInputChannels={self.max_in_ch}")
        print(f"[Open] channels={self.desired_channels}, rate={RATE}")
        
        self.stream = self.p.open(
            format=FORMAT,
            channels=self.desired_channels,
            rate=RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK
        )
        
    def _find_respeaker_input_device(self) -> Tuple[int, int]:
        """ReSpeaker 입력 장치 찾기"""
        device_index = None
        max_in_ch = 0
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            name = info.get('name', '')
            is_input = int(info.get('maxInputChannels', 0)) > 0
            if is_input and any(k.lower() in name.lower() for k in TARGET_DEVICE_KEYWORDS):
                device_index = i
                max_in_ch = int(info.get('maxInputChannels', 0))
                break
        
        # 못 찾았으면 기본 입력 장치 사용
        if device_index is None:
            default_idx = self.p.get_default_input_device_info().get('index', None)
            if default_idx is None:
                print("입력 장치를 찾을 수 없습니다.", file=sys.stderr)
                sys.exit(1)
            info = self.p.get_device_info_by_index(default_idx)
            device_index = default_idx
            max_in_ch = int(info.get('maxInputChannels', 0))
        
        return device_index, max_in_ch
    
    def _to_mono_int16(self, interleaved: np.ndarray, num_channels: int) -> np.ndarray:
        """멀티채널 int16 interleaved -> 모노 int16 (Channel 0만 사용)"""
        if num_channels <= 1:
            return interleaved.astype(np.int16)

        # 길이가 채널 수로 딱 나눠떨어지도록 잘라서 reshape
        usable_len = (len(interleaved) // num_channels) * num_channels
        if usable_len != len(interleaved):
            interleaved = interleaved[:usable_len]
        x = interleaved.reshape(-1, num_channels)

        # Channel 0만 사용 (ReSpeaker USB Mic Array의 후처리된 오디오)
        mono = x[:, 0].astype(np.int16)
        
        return mono

    def _calculate_db_level(self, interleaved: np.ndarray, num_channels: int) -> float:
        """dB 레벨 계산 (Channel 0만 사용)"""
        if num_channels <= 1:
            audio_data = interleaved.astype(np.float32)
        else:
            usable_len = (len(interleaved) // num_channels) * num_channels
            x = interleaved[:usable_len].reshape(-1, num_channels)

            # Channel 0만 사용 (ReSpeaker USB Mic Array의 후처리된 오디오)
            audio_data = x[:, 0].astype(np.float32)

        # RMS 계산
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Debug output removed
        
        if rms == 0:
            return -np.inf
        
        # dB 변환 (20 * log10(rms))
        # RMS가 0이 아닌 경우에만 dB 계산
        if rms > 0:
            db = 20 * np.log10(rms)
            
            # 유효한 dB 값인지 확인
            if np.isnan(db) or np.isinf(db):
                return -np.inf
                
            return db
        else:
            return -np.inf

    def _level_for_trigger(self, interleaved: np.ndarray, num_channels: int) -> float:
        """트리거 판정 레벨(RMS 또는 abs max) - Channel 0만 사용"""
        if num_channels <= 1:
            return float(np.max(np.abs(interleaved)))
        usable_len = (len(interleaved) // num_channels) * num_channels
        x = interleaved[:usable_len].reshape(-1, num_channels)

        # Channel 0만 사용 (ReSpeaker USB Mic Array의 후처리된 오디오)
        ch = x[:, 0].astype(np.int16)

        return float(np.max(np.abs(ch)))

    def start_monitoring(self) -> Optional[str]:
        """
        Start sound monitoring
        
        Returns:
            Recorded file path (when triggered), None (no trigger)
        """
        print("Waiting for sounds above 100dB... (Press Ctrl+C to stop)")
        
        recording = False
        frames_bytes = []
        samples_collected = 0
        target_samples = int(RECORD_SECONDS * RATE)
        
        try:
            # Monitor with a reasonable timeout
            import time
            start_time = time.time()
            timeout = 300  # 5 minutes timeout (longer for testing)
            
            while time.time() - start_time < timeout:
                raw = self.stream.read(CHUNK, exception_on_overflow=False)
                data_i16 = np.frombuffer(raw, dtype=np.int16)

                # Calculate dB level
                db_level = self._calculate_db_level(data_i16, self.desired_channels)
                
                # Check trigger (above 100dB)
                if not recording and db_level >= THRESHOLD_DB:
                    # 동시 녹음 제한 확인
                    with self.recording_lock:
                        if self.active_recordings >= self.max_concurrent_recordings:
                            print(f"Sound detected! ({db_level:.1f}dB) but max concurrent recordings reached ({self.max_concurrent_recordings})")
                            continue
                        
                        self.active_recordings += 1
                        print(f"Sound detected! ({db_level:.1f}dB) micarray wake up...")
                        print(f"Active recordings: {self.active_recordings}/{self.max_concurrent_recordings}")
                    
                    # Execute wake up if LED controller is available
                    if self.led_controller:
                        self.led_controller.wakeup_from_sleep()
                    
                    print("Recording started...")
                    recording = True
                    frames_bytes = []
                    samples_collected = 0

                if recording:
                    mono = self._to_mono_int16(data_i16, self.desired_channels)
                    
                    frames_bytes.append(mono.tobytes())
                    samples_collected += len(mono)

                    if samples_collected >= target_samples:
                        recording = False
                        print("녹음 종료. 파일 저장 중...")

                        # 파일명에 타임스탬프 포함
                        timestamp = int(time.time())
                        output_filename = os.path.join(self.output_dir, f"triggered_recording_{timestamp}.wav")
                        
                        # WAV 파일 저장
                        all_frames_data = b''.join(frames_bytes)
                        
                        wf = wave.open(output_filename, 'wb')
                        wf.setnchannels(1)
                        wf.setsampwidth(self.p.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(all_frames_data)
                        wf.close()

                        # 동시 녹음 카운터 감소
                        with self.recording_lock:
                            self.active_recordings -= 1
                            print(f"Recording completed. Active recordings: {self.active_recordings}/{self.max_concurrent_recordings}")

                        print(f"Saved: {output_filename}")
                        print("Waiting for sounds above 100dB...")
                        
                        return output_filename
            
            # Timeout reached
            print("Monitoring timeout (5 minutes) - no sound detected")
            return None

        except KeyboardInterrupt:
            print("\nMonitoring stopped...")
            return None
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        finally:
            if self.p:
                self.p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def main():
    """테스트용 메인 함수"""
    with SoundTrigger() as trigger:
        recorded_file = trigger.start_monitoring()
        if recorded_file:
            print(f"녹음 완료: {recorded_file}")


if __name__ == "__main__":
    main()
