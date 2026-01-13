#!/usr/bin/env python3
"""
Real-time visualization for Alpamayo reasoning and trajectory
"""

import pygame
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
from collections import deque
import threading
import queue

class AlpamayoVisualizer:
    """Real-time visualization of Alpamayo's reasoning and predictions"""
    
    def __init__(self, width=1920, height=1080):
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Alpamayo VLA - Chain of Causation Reasoning")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Data buffers
        self.camera_images = {}
        self.trajectory_buffer = deque(maxlen=100)
        self.reasoning_buffer = deque(maxlen=5)
        self.metrics_buffer = deque(maxlen=100)
        
        # Colors
        self.colors = {
            'background': (20, 20, 30),
            'panel': (40, 40, 50),
            'text': (200, 200, 200),
            'trajectory': (0, 255, 0),
            'reasoning': (100, 150, 255),
            'metrics': (255, 150, 100),
            'danger': (255, 50, 50),
            'safe': (50, 255, 50)
        }
        
        self.running = True
        self.data_queue = queue.Queue()
    
    def update_data(self, data):
        """Update visualization data"""
        self.data_queue.put(data)
    
    def draw_camera_panel(self, x, y, w, h):
        """Draw multi-camera view panel"""
        pygame.draw.rect(self.screen, self.colors['panel'], (x, y, w, h), 2)
        
        # Title
        title = self.font.render("Multi-Camera Input (4x)", True, self.colors['text'])
        self.screen.blit(title, (x + 10, y + 10))
        
        # Draw 4 camera views in 2x2 grid
        cam_w = (w - 30) // 2
        cam_h = (h - 60) // 2
        
        cameras = ['front_wide', 'front_tele', 'cross_left', 'cross_right']
        positions = [(x + 10, y + 40), (x + cam_w + 20, y + 40),
                    (x + 10, y + cam_h + 50), (x + cam_w + 20, y + cam_h + 50)]
        
        for cam_name, pos in zip(cameras, positions):
            # Draw camera frame
            pygame.draw.rect(self.screen, self.colors['text'], 
                           (*pos, cam_w, cam_h), 1)
            
            # Camera label
            label = self.font_small.render(cam_name.replace('_', ' ').title(), 
                                          True, self.colors['text'])
            self.screen.blit(label, (pos[0] + 5, pos[1] + 5))
            
            # Draw camera image if available
            if cam_name in self.camera_images:
                img = self.camera_images[cam_name]
                # Resize and convert for pygame
                img_resized = cv2.resize(img, (cam_w - 10, cam_h - 30))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
                self.screen.blit(img_surface, (pos[0] + 5, pos[1] + 25))
    
    def draw_reasoning_panel(self, x, y, w, h):
        """Draw Chain-of-Causation reasoning panel"""
        pygame.draw.rect(self.screen, self.colors['panel'], (x, y, w, h), 2)
        
        # Title
        title = self.font.render("Chain-of-Causation Reasoning", True, self.colors['reasoning'])
        self.screen.blit(title, (x + 10, y + 10))
        
        # Draw reasoning traces
        y_offset = 40
        for i, reasoning in enumerate(self.reasoning_buffer):
            if isinstance(reasoning, dict):
                # Primary decision
                if 'primary' in reasoning:
                    text = f"➤ {reasoning['primary']}"
                    rendered = self.font_small.render(text, True, self.colors['text'])
                    self.screen.blit(rendered, (x + 20, y + y_offset))
                    y_offset += 25
                
                # Causal factors
                if 'factors' in reasoning:
                    for factor in reasoning['factors'][:3]:
                        text = f"  → {factor}"
                        rendered = self.font_small.render(text, True, 
                                                         (150, 150, 150))
                        self.screen.blit(rendered, (x + 40, y + y_offset))
                        y_offset += 20
            else:
                # Simple text reasoning
                lines = str(reasoning).split('\n')
                for line in lines[:5]:
                    if line.strip():
                        rendered = self.font_small.render(line[:80], True, 
                                                         self.colors['text'])
                        self.screen.blit(rendered, (x + 20, y + y_offset))
                        y_offset += 20
            
            y_offset += 10
            if y_offset > h - 20:
                break
    
    def draw_trajectory_panel(self, x, y, w, h):
        """Draw predicted trajectory visualization"""
        pygame.draw.rect(self.screen, self.colors['panel'], (x, y, w, h), 2)
        
        # Title
        title = self.font.render("Predicted Trajectory (6.4s @ 10Hz)", True, 
                                self.colors['trajectory'])
        self.screen.blit(title, (x + 10, y + 10))
        
        # Create bird's eye view
        bev_center_x = x + w // 2
        bev_center_y = y + h // 2
        
        # Draw ego vehicle
        ego_rect = pygame.Rect(bev_center_x - 10, bev_center_y - 20, 20, 40)
        pygame.draw.rect(self.screen, (255, 255, 255), ego_rect)
        
        # Draw predicted trajectory
        if len(self.trajectory_buffer) > 0:
            trajectory = self.trajectory_buffer[-1]
            
            if isinstance(trajectory, np.ndarray):
                # Scale and draw waypoints
                scale = 5  # pixels per meter
                
                for i in range(len(trajectory) - 1):
                    # Current and next waypoint
                    wp1 = trajectory[i]
                    wp2 = trajectory[i + 1]
                    
                    # Convert to screen coordinates
                    x1 = int(bev_center_x + wp1[0] * scale)
                    y1 = int(bev_center_y - wp1[1] * scale)  # Invert Y
                    x2 = int(bev_center_x + wp2[0] * scale)
                    y2 = int(bev_center_y - wp2[1] * scale)
                    
                    # Color based on time (fade over distance)
                    color_intensity = int(255 * (1 - i / len(trajectory)))
                    color = (0, color_intensity, 0)
                    
                    # Draw line segment
                    pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 3)
                    
                    # Draw waypoint
                    pygame.draw.circle(self.screen, color, (x1, y1), 4)
        
        # Add scale indicator
        scale_text = self.font_small.render("Scale: 1m = 5px", True, 
                                           self.colors['text'])
        self.screen.blit(scale_text, (x + 10, y + h - 25))
    
    def draw_metrics_panel(self, x, y, w, h):
        """Draw performance metrics panel"""
        pygame.draw.rect(self.screen, self.colors['panel'], (x, y, w, h), 2)
        
        # Title
        title = self.font.render("Performance Metrics", True, self.colors['metrics'])
        self.screen.blit(title, (x + 10, y + 10))
        
        # Current metrics
        metrics = {
            'Speed': '30.5 km/h',
            'Throttle': '0.45',
            'Brake': '0.00',
            'Steering': '-0.12',
            'FPS': '10.0',
            'Latency': '95ms',
            'GPU Memory': '18.5 GB',
            'AlpaSim Score': '0.72'
        }
        
        y_offset = 40
        for key, value in metrics.items():
            text = f"{key}: {value}"
            
            # Color code certain metrics
            color = self.colors['text']
            if key == 'Speed' and float(value.split()[0]) > 50:
                color = self.colors['danger']
            elif key == 'FPS' and float(value.split()[0]) < 5:
                color = self.colors['danger']
            
            rendered = self.font_small.render(text, True, color)
            self.screen.blit(rendered, (x + 20, y + y_offset))
            y_offset += 25
        
        # Mini graph for speed history
        if len(self.metrics_buffer) > 1:
            graph_y = y + h - 100
            graph_h = 80
            graph_w = w - 40
            
            # Draw graph background
            pygame.draw.rect(self.screen, (30, 30, 40), 
                           (x + 20, graph_y, graph_w, graph_h), 0)
            
            # Plot speed history
            points = []
            for i, metric in enumerate(self.metrics_buffer):
                if 'speed' in metric:
                    px = x + 20 + int(i * graph_w / len(self.metrics_buffer))
                    py = graph_y + graph_h - int(metric['speed'] * graph_h / 60)
                    points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.colors['metrics'], 
                                False, points, 2)
    
    def draw_control_hints(self):
        """Draw control hints"""
        hints = [
            "Controls:",
            "Space - Emergency Stop",
            "R - Reset Scenario",
            "V - Change View",
            "T - Toggle Trajectory",
            "C - Show Reasoning",
            "ESC - Exit"
        ]
        
        y_offset = self.height - 180
        for hint in hints:
            text = self.font_small.render(hint, True, (100, 100, 100))
            self.screen.blit(text, (20, y_offset))
            y_offset += 20
    
    def run(self):
        """Main visualization loop"""
        print("🎨 Alpamayo Visualizer Started")
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Process data queue
            while not self.data_queue.empty():
                data = self.data_queue.get()
                if 'cameras' in data:
                    self.camera_images = data['cameras']
                if 'trajectory' in data:
                    self.trajectory_buffer.append(data['trajectory'])
                if 'reasoning' in data:
                    self.reasoning_buffer.append(data['reasoning'])
                if 'metrics' in data:
                    self.metrics_buffer.append(data['metrics'])
            
            # Clear screen
            self.screen.fill(self.colors['background'])
            
            # Draw panels
            # Top: Camera views
            self.draw_camera_panel(10, 10, self.width - 20, 400)
            
            # Middle left: Reasoning
            self.draw_reasoning_panel(10, 420, (self.width - 30) // 2, 350)
            
            # Middle right: Trajectory
            self.draw_trajectory_panel((self.width - 30) // 2 + 20, 420, 
                                      (self.width - 30) // 2, 350)
            
            # Bottom: Metrics
            self.draw_metrics_panel(10, 780, self.width - 20, 280)
            
            # Control hints
            self.draw_control_hints()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS for visualization
        
        pygame.quit()
        print("🛑 Visualizer stopped")

def main():
    """Run standalone visualizer"""
    visualizer = AlpamayoVisualizer()
    
    # Generate demo data
    def generate_demo_data():
        while visualizer.running:
            # Demo reasoning
            reasoning = {
                'primary': 'Approaching intersection with traffic light',
                'factors': [
                    'Traffic light is green',
                    'No pedestrians detected',
                    'Clear path ahead'
                ]
            }
            
            # Demo trajectory (spiral for visualization)
            t = time.time()
            trajectory = np.array([
                [i * np.cos(t + i * 0.1), 
                 i * np.sin(t + i * 0.1), 
                 0] 
                for i in range(64)
            ])
            
            # Demo metrics
            metrics = {
                'speed': 30 + 10 * np.sin(t),
                'fps': 10.0
            }
            
            visualizer.update_data({
                'reasoning': reasoning,
                'trajectory': trajectory,
                'metrics': metrics
            })
            
            time.sleep(0.1)
    
    # Start demo data thread
    demo_thread = threading.Thread(target=generate_demo_data, daemon=True)
    demo_thread.start()
    
    # Run visualizer
    visualizer.run()

if __name__ == "__main__":
    main()