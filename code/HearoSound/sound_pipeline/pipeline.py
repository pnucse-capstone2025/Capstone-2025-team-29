import os
import sys
import time
import threading
import queue
import argparse
from typing import Optional, Dict, Any, Set
from datetime import datetime, timedelta


from sound_trigger import SoundTrigger
from doa_calculator import create_doa_calculator
from separator import create_single_separator
from led_controller import create_led_controller


class SingleSoundPipeline:
    def __init__(self, output_dir: str = "pipeline_output", 
                 model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
                 device: str = "auto", backend_url: str = "http://13.238.200.232:8000/sound-events/"):
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = device
        self.backend_url = backend_url
        
        os.makedirs(output_dir, exist_ok=True)
        self.is_running = False
        self.current_sent_classes: Set[str] = set()

        self.stats = {
            "total_detected": 0,
            "successful_separations": 0,
            "backend_sends": 0,
            "led_activations": 0,
            "duplicate_skips": 0
        }
    
    def _initialize_components(self):
        """Initialize components"""
        print("=== Single Thread Pipeline Initialization ===")
        
        print("1. Initializing Sound Trigger...")
        self.sound_trigger = SoundTrigger(os.path.join(self.output_dir, "recordings"), None)
        
        print("2. Initializing DOA Calculator...")
        self.doa_calculator = create_doa_calculator()
        
        print("3. Initializing LED Controller...")
        self.led_controller = create_led_controller()
        if self.led_controller is None:
            print("⚠️ LED Controller not available - LED control disabled")
        
        print("4. Initializing Single Separator...")
        self.sound_separator = create_single_separator(
            model_name=self.model_name, 
            backend_url=self.backend_url, 
            led_controller=self.led_controller  
        )
        
        if hasattr(self.sound_separator, 'ast_processor') and hasattr(self.sound_separator.ast_processor, 'is_available'):
            if self.sound_separator.ast_processor.is_available:
                print("✅ Single Separator initialized successfully")
            else:
                print("❌ Single Separator AST model initialization failed!")
        else:
            print("⚠️ Single Separator status unknown")
        
        print("=== Single Thread Pipeline Ready ===")
    
    def _main_loop(self):
        """Main loop - sequential processing from sound detection to separation"""
        while self.is_running:
            try:
                recorded_file = self.sound_trigger.start_monitoring()
                
                if recorded_file and self.is_running:
                    self.stats["total_detected"] += 1
                    print(f"\n🎵 Processing: {os.path.basename(recorded_file)}")
                    
                    recording_end_time = datetime.utcnow()
                    separation_result = self._process_separation(recorded_file, recording_end_time)
                    
                    if separation_result["success"]:
                        separated_sources = separation_result.get("separated_sources", [])
                        print(f"✅ Separation completed: {len(separated_sources)} sources")
                    else:
                        print(f"❌ Separation failed: {separation_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                continue
    
    
    def _process_separation(self, audio_file: str, recording_end_time: datetime) -> Dict[str, Any]:
        """Audio source separation and backend transmission for each pass (preventing duplicate class transmission)"""
        try:
            # 1. Calculate DOA
            angle = self.doa_calculator.get_direction_with_retry(max_retries=2)
            if angle is None:
                angle = 0
            
            print(f"📍 Direction: {angle}°")
            
            # 2. Perform audio source separation (using separator)
            print("🔍 Starting source separation...")
            
            # Initialize duplicate class set for current audio file
            self.current_sent_classes.clear()
            
            # Execute separation using single_separator
            separated_sources = self.sound_separator.separate_and_process(
                audio_file, 
                angle=angle, 
                max_passes=2,
                output_dir=self.output_dir
            )
            
            if separated_sources:
                print(f"✅ Separation completed: {len(separated_sources)} sources")
                self.stats["successful_separations"] += 1
                
                # Update backend transmission and LED activation statistics
                for source in separated_sources:
                    class_name = source['class_name']
                    sound_type = source['sound_type']
                    
                    # Check duplicate class
                    if class_name in self.current_sent_classes:
                        print(f"⏭️ SKIP: {class_name} ({sound_type}) - Duplicate")
                        self.stats["duplicate_skips"] += 1
                        continue
                    
                    # Backend transmission statistics
                    if source.get('backend_sent', False):
                        self.stats["backend_sends"] += 1
                    
                    # LED activation statistics
                    if source.get('led_activated', False):
                        self.stats["led_activations"] += 1
                    
                    # Record transmitted class for current audio file
                    self.current_sent_classes.add(class_name)
                    
                    # Simplified output
                    backend_status = "✅" if source.get('backend_sent', False) else "❌"
                    led_status = "💡" if source.get('led_activated', False) else "⭕"
                    print(f"🎵 {class_name} ({sound_type}) - Backend: {backend_status}, LED: {led_status}")
                
                return {"success": True, "separated_sources": separated_sources}
            else:
                print("❌ No sources separated (Silence detected or no valid sounds)")
                return {"success": False, "error": "No sources separated"}
                
        except Exception as e:
            print(f"❌ Separation error: {e}")
            return {"success": False, "error": str(e)}
    
    
    def start(self):
        """Start pipeline - sequential execution in one thread"""
        if self.is_running:
            print("⚠️ Pipeline is already running")
            return
        
        print("🚀 Starting Single Thread Sound Pipeline...")
        print("=" * 60)
        print("Mode: Sound Detection → Source Separation → Backend/LED")
        print("=" * 60)
        
        # Initialize components
        self._initialize_components()
        
        # Start main loop
        self.is_running = True
        
        print("\n✅ Single Thread Sound Pipeline started successfully!")
        print("📡 Monitoring for sounds above 100dB...")
        print("🔍 Will process audio separation with type filtering")
        print("📤 Backend: Only danger/help/warning types will be sent")
        print("💡 LED: Only danger/help/warning types will activate LED")
        print("⏭️ Will skip 'other' types and duplicate classes")
        print("🖥️ Optimized for Raspberry Pi 5 performance")
        print("\nPress Ctrl+C to stop")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n🛑 Stopping pipeline...")
            self.stop()
    
    def stop(self):
        """Stop pipeline"""
        if not self.is_running:
            print("⚠️ Pipeline is not running")
            return
        
        print("🛑 Stopping Single Thread Sound Pipeline...")
        
        # Turn off LED
        if self.led_controller:
            self.led_controller.turn_off()
        
        self.is_running = False
        print("✅ Single Thread Sound Pipeline stopped")
        self._print_statistics()
    
    def _print_statistics(self):
        """Print statistics"""
        print("\n=== Single Thread Pipeline Statistics ===")
        print(f"Total detected: {self.stats['total_detected']}")
        print(f"Successful separations: {self.stats['successful_separations']}")
        print(f"Backend sends: {self.stats['backend_sends']}")
        print(f"LED activations: {self.stats['led_activations']}")
        print(f"Duplicate skips: {self.stats['duplicate_skips']}")
        print("==========================================\n")
    
    def cleanup(self):
        """Resource cleanup"""
        if self.is_running:
            self.stop()
        
        # Component cleanup
        if hasattr(self, 'sound_trigger') and self.sound_trigger:
            self.sound_trigger.cleanup()
        if hasattr(self, 'doa_calculator') and self.doa_calculator:
            self.doa_calculator.cleanup()
        if hasattr(self, 'sound_separator') and self.sound_separator:
            self.sound_separator.cleanup()
        if hasattr(self, 'led_controller') and self.led_controller:
            self.led_controller.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Single Thread Sound Pipeline - Raspberry Pi 5 Optimized with Type Filtering")
    parser.add_argument("--output", "-o", default="pipeline_output", help="Output directory")
    parser.add_argument("--model", "-m", default="MIT/ast-finetuned-audioset-10-10-0.4593", help="AST model name")
    parser.add_argument("--device", "-d", default="auto", help="Device (auto/cpu/cuda) - CPU optimized for RPi5")
    parser.add_argument("--backend-url", default="http://13.238.200.232:8000/sound-events/", help="Backend API URL")
    
    args = parser.parse_args()
    
    print("🎵 Single Thread Sound Pipeline v2.0 - Raspberry Pi 5 Edition")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device} (CPU optimized for Raspberry Pi 5)")
    print(f"Backend URL: {args.backend_url}")
    print("🔍 Sound Type Filtering: Only danger/help/warning types sent to backend")
    print("💡 LED Control: Activated for danger/help/warning types only")
    print("=" * 60)
    
    # Execute pipeline
    with SingleSoundPipeline(
        output_dir=args.output,
        model_name=args.model,
        device=args.device,
        backend_url=args.backend_url
    ) as pipeline:
        pipeline.start()


if __name__ == "__main__":
    main()

