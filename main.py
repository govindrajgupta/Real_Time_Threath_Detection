import cv2
import time
import argparse
from cvzone.PoseModule import PoseDetector
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HumanDetectionSystem:
    def __init__(self, confidence=0.5, camera_id=0, resolution=(640, 480), 
                 detection_threshold=50, show_video=True):
        """
        Initialize the human detection system
        
        Args:
            confidence: Detection confidence threshold (0-1)
            camera_id: Camera device ID
            resolution: Video resolution as tuple (width, height)
            detection_threshold: Number of positive detections before alert
            show_video: Whether to display video output
        """
        self.detector = PoseDetector(detectionCon=confidence)
        self.camera_id = camera_id
        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.show_video = show_video
        self.detection_count = 0
        self.alert_sent = False
        self.running = False
        self.cap = None
        
    def send_alert(self):
        """Send alert when human presence is consistently detected"""
        try:
            # Import here to avoid issues if module is missing
            import send
            logger.info("Sending alert notification")
            send.sendSms()
            return True
        except ImportError:
            logger.error("Could not import 'send' module. Alert not sent.")
            return False
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
            return False

    def start(self):
        """Initialize and start the detection system"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera_id}")
                
            # Set resolution
            self.cap.set(3, self.resolution[0])
            self.cap.set(4, self.resolution[1])
            
            self.running = True
            self.detection_count = 0
            self.alert_sent = False
            
            logger.info(f"Detection system started - Camera: {self.camera_id}, "
                       f"Resolution: {self.resolution}, "
                       f"Threshold: {self.detection_threshold}")
            
            self._process_frames()
            
        except Exception as e:
            logger.error(f"Error starting detection system: {str(e)}")
            self.stop()
    
    def stop(self):
        """Stop the detection system and release resources"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Detection system stopped")
    
    def _process_frames(self):
        """Main processing loop for video frames"""
        fps_timer = time.time()
        frame_count = 0
        
        while self.running:
            try:
                success, img = self.cap.read()
                if not success:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                # Detect pose
                img = self.detector.findPose(img, draw=self.show_video)
                lmlist, bbox = self.detector.findPosition(img, draw=self.show_video)
                
                # Calculate FPS
                frame_count += 1
                if time.time() - fps_timer >= 1:
                    fps = frame_count
                    frame_count = 0
                    fps_timer = time.time()
                    if self.show_video:
                        cv2.putText(img, f"FPS: {fps}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Handle detection
                if len(lmlist) > 0:
                    self.detection_count += 1
                    status_text = f"Human Detected: {self.detection_count}/{self.detection_threshold}"
                    logger.debug(status_text)
                    
                    if self.show_video:
                        # Display detection counter
                        cv2.putText(img, status_text, (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Send alert if threshold reached and not already sent
                    if self.detection_count >= self.detection_threshold and not self.alert_sent:
                        self.alert_sent = True
                        # Send alert in separate thread to avoid blocking
                        threading.Thread(target=self.send_alert).start()
                else:
                    # Gradually decrease count when no detection (prevents false negatives)
                    self.detection_count = max(0, self.detection_count - 0.5)
                
                # Show video if enabled
                if self.show_video:
                    cv2.imshow("Human Detection", img)
                    
                    # Check for quit key
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == 27:  # q or ESC
                        logger.info("User requested exit")
                        break
                        
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                
        self.stop()


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Human Detection System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--threshold", type=int, default=50, 
                        help="Detection threshold count before alert")
    parser.add_argument("--confidence", type=float, default=0.5, 
                        help="Detection confidence (0-1)")
    parser.add_argument("--no-display", action="store_true", 
                        help="Disable video display")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Create and start detection system
    detector = HumanDetectionSystem(
        camera_id=args.camera,
        resolution=(args.width, args.height),
        detection_threshold=args.threshold,
        confidence=args.confidence,
        show_video=not args.no_display
    )
    
    try:
        detector.start()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
