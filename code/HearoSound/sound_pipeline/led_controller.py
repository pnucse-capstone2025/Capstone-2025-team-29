#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED Controller Module
- led_control.py 기반으로 구현
- 분리 결과에 따른 LED 색상 출력
- danger=빨강(0xFF0000), warning=노랑(0xFFFF00), help=초록(0x00FF00)
"""

import usb.core
import usb.util
import time
import sys
import os
from typing import Optional, Dict, Any

# usb_pixel_ring_v2.py 직접 import를 위한 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mic_array', 'pixel_ring', 'pixel_ring'))

try:
    from usb_pixel_ring_v2 import PixelRing, find
except ImportError:
    print("Warning: usb_pixel_ring_v2 module not found. LED control will be disabled.")
    PixelRing = None
    find = None

# LED 색상 정의
LED_COLORS = {
    "danger": 0xFF0000,    # 빨간색
    "warning": 0xFFFF00,   # 노란색
    "help": 0x00FF00,      # 초록색
    "default": 0x000000,   # 파란색 (기본값)
    "off": 0x000000        # 꺼짐
}

class LEDController:
    def __init__(self):
        """
        LED Controller 초기화
        """
        self.dev = None
        self.pixel_ring = None
        self.is_available = False
        self.current_color = LED_COLORS["off"]
        
        self._initialize_device()
    
    def _initialize_device(self):
        """ReSpeaker USB 장치 초기화 - 무조건 PixelRing 사용"""
        try:
            # PixelRing 모듈이 없으면 강제로 실패
            if not PixelRing or not find:
                print("[LED] PixelRing module not available - LED control disabled")
                self.is_available = False
                return
            
            # find 함수를 사용하여 PixelRing 인스턴스 생성
            self.pixel_ring = find()
            
            if self.pixel_ring:
                self.is_available = True
                print(f"[LED] ReSpeaker device found and initialized")
                
                # 초기 설정 - sleep 모드로 시작
                self.pixel_ring.set_brightness(0x001)
                self.pixel_ring.off()  # sleep 모드로 시작
                print("[LED] Started in sleep mode")
            else:
                print("[LED] ReSpeaker device not found")
                self.is_available = False
                
        except Exception as e:
            print(f"[LED] Initialization error: {e}")
            self.is_available = False
    
    def set_color(self, color: int, duration: float = 0.0) -> bool:
        """
        LED 색상 설정
        
        Args:
            color: RGB 색상 값 (0xRRGGBB)
            duration: 지속 시간 (초), 0이면 계속 켜짐
            
        Returns:
            설정 성공 여부
        """
        if not self.is_available or not self.pixel_ring:
            print(f"[LED] Device not available, cannot set color 0x{color:06X}")
            return False
        
        try:
            self.pixel_ring.mono(color)
            self.current_color = color
            print(f"[LED] Color set to 0x{color:06X}")
            
            if duration > 0:
                time.sleep(duration)
                self.pixel_ring.off()
                self.current_color = LED_COLORS["off"]
                print(f"[LED] Color turned off after {duration}s")
            
            return True
            
        except Exception as e:
            print(f"[LED] Error setting color: {e}")
            return False
    
    def set_sound_type_color(self, sound_type: str, duration: float = 5.0) -> bool:
        """
        소리 타입에 따른 LED 색상 설정 (기본 5초 유지)
        
        Args:
            sound_type: 소리 타입 (danger/warning/help/other)
            duration: 지속 시간 (초), 기본값 5초
            
        Returns:
            설정 성공 여부
        """
        color = LED_COLORS.get(sound_type, LED_COLORS["default"])
        
        # danger 타입인 경우 10초간 깜빡임
        if sound_type == "danger":
            return self.blink_sound_type(sound_type, blink_count=10, blink_duration=1.0)
        else:
            return self.set_color(color, duration)
    
    def blink_color(self, color: int, blink_count: int = 3, blink_duration: float = 0.5) -> bool:
        """
        LED 깜빡이기
        
        Args:
            color: RGB 색상 값 (0xRRGGBB)
            blink_count: 깜빡임 횟수
            blink_duration: 깜빡임 간격 (초)
            
        Returns:
            설정 성공 여부
        """
        if not self.is_available or not self.pixel_ring:
            print(f"[LED] Device not available, cannot blink color 0x{color:06X}")
            return False
        
        try:
            for i in range(blink_count):
                self.pixel_ring.mono(color)
                time.sleep(blink_duration)
                self.pixel_ring.off()
                if i < blink_count - 1:  # 마지막이 아니면
                    time.sleep(blink_duration)
            
            self.current_color = LED_COLORS["off"]
            print(f"[LED] Blinked color 0x{color:06X} {blink_count} times")
            return True
            
        except Exception as e:
            print(f"[LED] Error blinking color: {e}")
            return False
    
    def blink_sound_type(self, sound_type: str, blink_count: int = 3, blink_duration: float = 0.5) -> bool:
        """
        소리 타입에 따른 LED 깜빡이기
        
        Args:
            sound_type: 소리 타입 (danger/warning/help/other)
            blink_count: 깜빡임 횟수
            blink_duration: 깜빡임 간격 (초)
            
        Returns:
            설정 성공 여부
        """
        color = LED_COLORS.get(sound_type, LED_COLORS["default"])
        return self.blink_color(color, blink_count, blink_duration)
    
    def wakeup_animation(self) -> bool:
        """
        깨어남 애니메이션 실행
        
        Returns:
            실행 성공 여부
        """
        if not self.is_available or not self.pixel_ring:
            print("[LED] Device not available, cannot run wakeup animation")
            return False
        
        try:
            self.pixel_ring.wakeup(180)
            print("[LED] Wakeup animation completed")
            return True
            
        except Exception as e:
            print(f"[LED] Error running wakeup animation: {e}")
            return False
    
    def wakeup_from_sleep(self) -> bool:
        """
        sleep 모드에서 깨어나기
        
        Returns:
            깨어나기 성공 여부
        """
        if not self.is_available or not self.pixel_ring:
            print("[LED] Device not available, cannot wakeup from sleep")
            return False
        
        try:
            self.pixel_ring.wakeup(180)
            print("[LED] Woke up from sleep mode")
            return True
            
        except Exception as e:
            print(f"[LED] Error waking up from sleep: {e}")
            return False
    
    def listen_animation(self) -> bool:
        """
        듣기 애니메이션 실행
        
        Returns:
            실행 성공 여부
        """
        if not self.is_available or not self.pixel_ring:
            print("[LED] Device not available, cannot run listen animation")
            return False
        
        try:
            self.pixel_ring.listen()
            print("[LED] Listen animation completed")
            return True
            
        except Exception as e:
            print(f"[LED] Error running listen animation: {e}")
            return False
    
    def activate_led(self, angle: int, class_name: str, sound_type: str) -> bool:
        """
        각도와 소리 정보에 따라 LED 활성화 (각도 기반)
        
        Args:
            angle: 소리 방향 각도 (0-359)
            class_name: 분류된 소리 클래스명
            sound_type: 소리 타입 (danger/warning/help/other)
            
        Returns:
            활성화 성공 여부
        """
        if not self.is_available or not self.pixel_ring:
            print(f"[LED] Device not available, cannot activate LED for {class_name} ({sound_type})")
            return False
        
        try:
            # danger, warning, help 타입만 각도 기반 LED 제어
            if sound_type in ["danger", "warning", "help"]:
                print(f"[LED] {sound_type.upper()} detected: {class_name} at {angle}° - activating directional LED")
                return self.set_directional_led(angle, sound_type, class_name)
            else:
                print(f"[LED] {sound_type.upper()} detected: {class_name} - using default LED")
                return self.set_sound_type_color(sound_type, duration=5.0)
            
        except Exception as e:
            print(f"[LED] Error activating LED: {e}")
            return False
    
    def set_directional_led(self, angle: int, sound_type: str, class_name: str) -> bool:
        """
        각도 기반 방향성 LED 제어 (가장 가까운 LED + 양옆 LED)
        
        Args:
            angle: 소리 방향 각도 (0-359)
            sound_type: 소리 타입 (danger/warning/help)
            class_name: 분류된 소리 클래스명
            
        Returns:
            설정 성공 여부
        """
        if not self.is_available or not self.pixel_ring:
            print(f"[LED] Device not available, cannot set directional LED")
            return False
        
        try:
            # 12개 LED 중 가장 가까운 LED 인덱스 계산
            led_index = self._angle_to_led_index(angle)
            
            # 양옆 LED 인덱스 계산
            left_led = (led_index - 1) % 12
            right_led = (led_index + 1) % 12
            
            # 색상 가져오기
            color = LED_COLORS.get(sound_type, LED_COLORS["default"])
            
            # RGB 값 추출
            r = (color >> 16) & 0xFF
            g = (color >> 8) & 0xFF
            b = color & 0xFF
            
            print(f"[LED] Directional LED: angle {angle}° -> LED {led_index} (left: {left_led}, right: {right_led})")
            print(f"[LED] Color: {sound_type} (R:{r:02X}, G:{g:02X}, B:{b:02X})")
            
            # 12개 LED 배열 생성 (모두 꺼진 상태)
            led_colors = [0, 0, 0, 0] * 12  # [r, g, b, 0] * 12
            
            # 선택된 3개 LED만 색상 설정
            for led_idx in [left_led, led_index, right_led]:
                led_colors[led_idx * 4] = r      # Red
                led_colors[led_idx * 4 + 1] = g  # Green
                led_colors[led_idx * 4 + 2] = b  # Blue
                led_colors[led_idx * 4 + 3] = 0  # Reserved
            
            # Custom mode로 LED 설정 (Command 6)
            self.pixel_ring.customize(led_colors)
            
            # danger 타입인 경우 깜빡임 효과
            if sound_type == "danger":
                return self._blink_directional_led(led_colors, blink_count=10, blink_duration=1.0)
            else:
                # 5초 후 자동으로 끄기
                import threading
                def turn_off_after_delay():
                    time.sleep(5.0)
                    self.turn_off()
                
                threading.Thread(target=turn_off_after_delay, daemon=True).start()
                return True
            
        except Exception as e:
            print(f"[LED] Error setting directional LED: {e}")
            return False
    
    def _angle_to_led_index(self, angle: int) -> int:
        """
        각도를 LED 인덱스로 변환 (0-11)
        
        Args:
            angle: 각도 (0-359)
            
        Returns:
            LED 인덱스 (0-11)
        """
        # 각도를 0-359 범위로 정규화
        angle = angle % 360
        
        # 12개 LED로 나누어 인덱스 계산 (30도씩)
        led_index = int(round(angle / 30.0)) % 12
        
        return led_index
    
    def _blink_directional_led(self, led_colors: list, blink_count: int = 10, blink_duration: float = 1.0) -> bool:
        """
        방향성 LED 깜빡이기
        
        Args:
            led_colors: LED 색상 배열
            blink_count: 깜빡임 횟수
            blink_duration: 깜빡임 간격 (초)
            
        Returns:
            설정 성공 여부
        """
        if not self.is_available or not self.pixel_ring:
            return False
        
        try:
            for i in range(blink_count):
                # LED 켜기
                self.pixel_ring.customize(led_colors)
                time.sleep(blink_duration)
                
                # LED 끄기
                self.turn_off()
                if i < blink_count - 1:  # 마지막이 아니면
                    time.sleep(blink_duration)
            
            print(f"[LED] Directional LED blinked {blink_count} times")
            return True
            
        except Exception as e:
            print(f"[LED] Error blinking directional LED: {e}")
            return False
    
    def turn_off(self) -> bool:
        """
        LED 끄기
        
        Returns:
            끄기 성공 여부
        """
        if not self.is_available or not self.pixel_ring:
            print("[LED] Device not available, cannot turn off")
            return False
        
        try:
            self.pixel_ring.off()
            self.current_color = LED_COLORS["off"]
            print("[LED] Turned off")
            return True
            
        except Exception as e:
            print(f"[LED] Error turning off: {e}")
            return False
    
    def get_current_color(self) -> int:
        """
        현재 LED 색상 가져오기
        
        Returns:
            현재 색상 값
        """
        return self.current_color
    
    def is_device_available(self) -> bool:
        """
        LED 장치 사용 가능 여부 확인
        
        Returns:
            True (사용 가능), False (사용 불가)
        """
        return self.is_available
    
    def cleanup(self):
        """리소스 정리"""
        if self.is_available and self.pixel_ring:
            try:
                self.pixel_ring.off()
            except:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class MockLEDController:
    """
    LED 장치가 없을 때 사용하는 Mock 클래스
    콘솔에 색상 정보 출력
    """
    
    def __init__(self):
        self.is_available = False
        self.current_color = LED_COLORS["off"]
        print("[LED] Mock LED Controller initialized (no real device)")
    
    def set_color(self, color: int, duration: float = 0.0) -> bool:
        """Mock 색상 설정"""
        self.current_color = color
        print(f"[LED] Mock: Color set to 0x{color:06X}")
        if duration > 0:
            print(f"[LED] Mock: Color will turn off after {duration}s")
        return True
    
    def set_sound_type_color(self, sound_type: str, duration: float = 5.0) -> bool:
        """Mock 소리 타입 색상 설정 (기본 5초 유지)"""
        color = LED_COLORS.get(sound_type, LED_COLORS["default"])
        
        # danger 타입인 경우 10초간 깜빡임
        if sound_type == "danger":
            print(f"[LED] Mock: Sound type '{sound_type}' -> 10초간 깜빡임")
            return self.blink_sound_type(sound_type, blink_count=10, blink_duration=1.0)
        else:
            print(f"[LED] Mock: Sound type '{sound_type}' -> Color 0x{color:06X} for {duration}s")
            return self.set_color(color, duration)
    
    def blink_color(self, color: int, blink_count: int = 3, blink_duration: float = 0.5) -> bool:
        """Mock 깜빡이기"""
        print(f"[LED] Mock: Blinking color 0x{color:06X} {blink_count} times")
        for i in range(blink_count):
            print(f"[LED] Mock: Blink {i+1}/{blink_count}")
            time.sleep(blink_duration)
        return True
    
    def blink_sound_type(self, sound_type: str, blink_count: int = 3, blink_duration: float = 0.5) -> bool:
        """Mock 소리 타입 깜빡이기"""
        color = LED_COLORS.get(sound_type, LED_COLORS["default"])
        print(f"[LED] Mock: Blinking sound type '{sound_type}' (0x{color:06X}) {blink_count} times")
        return self.blink_color(color, blink_count, blink_duration)
    
    def wakeup_animation(self) -> bool:
        """Mock 깨어남 애니메이션"""
        print("[LED] Mock: Wakeup animation")
        return True
    
    def wakeup_from_sleep(self) -> bool:
        """Mock sleep 모드에서 깨어나기"""
        print("[LED] Mock: Woke up from sleep mode")
        return True
    
    def listen_animation(self) -> bool:
        """Mock 듣기 애니메이션"""
        print("[LED] Mock: Listen animation")
        return True
    
    def activate_led(self, angle: int, class_name: str, sound_type: str) -> bool:
        """Mock LED 활성화 (각도 기반)"""
        color = LED_COLORS.get(sound_type, LED_COLORS["default"])
        
        # danger, warning, help 타입만 각도 기반 LED 제어
        if sound_type in ["danger", "warning", "help"]:
            print(f"[LED] Mock: {sound_type.upper()} detected: {class_name} at {angle}° - activating directional LED")
            return self.set_directional_led(angle, sound_type, class_name)
        else:
            print(f"[LED] Mock: {sound_type.upper()} detected: {class_name} - using default LED")
            return self.set_sound_type_color(sound_type, duration=5.0)
    
    def set_directional_led(self, angle: int, sound_type: str, class_name: str) -> bool:
        """Mock 방향성 LED 제어"""
        # 12개 LED 중 가장 가까운 LED 인덱스 계산
        led_index = self._angle_to_led_index(angle)
        
        # 양옆 LED 인덱스 계산
        left_led = (led_index - 1) % 12
        right_led = (led_index + 1) % 12
        
        # 색상 가져오기
        color = LED_COLORS.get(sound_type, LED_COLORS["default"])
        
        # RGB 값 추출
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        
        print(f"[LED] Mock: Directional LED: angle {angle}° -> LED {led_index} (left: {left_led}, right: {right_led})")
        print(f"[LED] Mock: Color: {sound_type} (R:{r:02X}, G:{g:02X}, B:{b:02X})")
        
        if sound_type == "danger":
            print(f"[LED] Mock: DANGER - blinking directional LED 10 times")
            return self._blink_directional_led_mock(led_index, left_led, right_led, color, blink_count=10, blink_duration=1.0)
        else:
            print(f"[LED] Mock: {sound_type.upper()} - directional LED for 5 seconds")
            return True
    
    def _angle_to_led_index(self, angle: int) -> int:
        """Mock 각도를 LED 인덱스로 변환"""
        angle = angle % 360
        led_index = int(round(angle / 30.0)) % 12
        return led_index
    
    def _blink_directional_led_mock(self, led_index: int, left_led: int, right_led: int, color: int, blink_count: int = 10, blink_duration: float = 1.0) -> bool:
        """Mock 방향성 LED 깜빡이기"""
        print(f"[LED] Mock: Blinking LEDs {left_led}, {led_index}, {right_led} {blink_count} times")
        for i in range(blink_count):
            print(f"[LED] Mock: Blink {i+1}/{blink_count} - LEDs {left_led}, {led_index}, {right_led} ON")
            time.sleep(blink_duration)
            print(f"[LED] Mock: Blink {i+1}/{blink_count} - LEDs OFF")
            if i < blink_count - 1:
                time.sleep(blink_duration)
        return True
    
    def turn_off(self) -> bool:
        """Mock 끄기"""
        self.current_color = LED_COLORS["off"]
        print("[LED] Mock: Turned off")
        return True
    
    def get_current_color(self) -> int:
        """Mock 현재 색상"""
        return self.current_color
    
    def is_device_available(self) -> bool:
        """Mock은 항상 사용 가능"""
        return True
    
    def cleanup(self):
        """Mock 정리"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def create_led_controller() -> LEDController:
    """
    LED Controller 인스턴스 생성 - 무조건 실제 PixelRing 사용
    
    Returns:
        LEDController 인스턴스 (실패 시 None)
    """
    controller = LEDController()
    
    if not controller.is_device_available():
        print("[LED] PixelRing not available - LED control disabled")
        return None
    
    return controller


def main():
    """테스트용 메인 함수"""
    with create_led_controller() as led:
        print("LED Controller 테스트 시작...")
        
        # 깨어남 애니메이션
        led.wakeup_animation()
        time.sleep(1)
        
        # 듣기 애니메이션
        led.listen_animation()
        time.sleep(1)
        
        # 각 소리 타입별 색상 테스트
        sound_types = ["danger", "warning", "help", "other"]
        
        for sound_type in sound_types:
            print(f"\nTesting {sound_type}...")
            
            # 색상 설정
            led.set_sound_type_color(sound_type, duration=1.0)
            time.sleep(0.5)
            
            # 깜빡이기
            led.blink_sound_type(sound_type, blink_count=2, blink_duration=0.3)
            time.sleep(0.5)
        
        # 끄기
        led.turn_off()
        
        print("\nLED Controller 테스트 완료")


if __name__ == "__main__":
    main()

