import socket
import threading
import queue
import time
import numpy as np
import cv2
import base64
import json
from PIL import Image
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import matplotlib.pyplot as plt
import io
from typing import Dict, List, Optional, Any

class TetrisStreamHandler(BaseHTTPRequestHandler):
    """HTTP Handler for streaming Tetris frames"""
    
    def do_GET(self):
        if self.path == '/':
            # Serve HTML page
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Tetris Worker {self.server.worker_id} Stream</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                    #stats {{ margin-top: 10px; }}
                </style>
                <script>
                    function updateStream() {{
                        const img = document.getElementById('stream');
                        const statsDiv = document.getElementById('stats');
                        
                        fetch('/stream')
                            .then(response => response.json())
                            .then(data => {{
                                img.src = 'data:image/jpeg;base64,' + data.image;
                                statsDiv.innerHTML = `
                                    <p>Worker: {self.server.worker_id}</p>
                                    <p>FPS: ${{data.fps.toFixed(1)}}</p>
                                    <p>Time: ${{new Date().toLocaleTimeString()}}</p>
                                `;
                            }})
                            .catch(err => console.error(err));
                    }}
                    
                    // Update every 100ms (10 FPS max)
                    setInterval(updateStream, 100);
                </script>
            </head>
            <body>
                <h1>Tetris Worker {self.server.worker_id} Stream</h1>
                <img id="stream" src="" alt="Tetris Stream" />
                <div id="stats"></div>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode())
            
        elif self.path == '/stream':
            # Serve the latest frame as JSON with base64 encoded image
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.end_headers()
            
            # Get the latest frame
            frame = self.server.get_current_frame()
            if frame is None:
                frame = np.zeros((512, 512), dtype=np.uint8)
            
            # Convert to JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                buffer = BytesIO()
                Image.fromarray(frame).save(buffer, format='JPEG', quality=80)
                buffer = buffer.getvalue()
            else:
                buffer = buffer.tobytes()
            
            # Encode as base64
            b64_encoded = base64.b64encode(buffer).decode('utf-8')
            
            # Create response with image and stats
            response = {
                'image': b64_encoded,
                'fps': self.server.current_fps,
                'worker_id': self.server.worker_id
            }
            
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Disable logging to prevent console spam
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    def __init__(self, server_address, RequestHandlerClass, worker_id=0):
        super().__init__(server_address, RequestHandlerClass)
        self.frame_queue = queue.Queue(maxsize=2)  # Keep only the most recent frame
        self.current_frame = None
        self.current_fps = 0
        self.worker_id = worker_id
        self.last_frame_time = time.time()
        self.frame_times = []
    
    def update_frame(self, frame):
        # Calculate FPS
        now = time.time()
        self.frame_times.append(now - self.last_frame_time)
        self.last_frame_time = now
        
        # Keep only the last 10 frame times for FPS calculation
        if len(self.frame_times) > 10:
            self.frame_times.pop(0)
        
        # Calculate FPS
        if self.frame_times:
            self.current_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        # Update current frame and add to queue
        self.current_frame = frame
        
        # Add to queue, replacing oldest frame if full
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def get_current_frame(self):
        return self.current_frame

class TetrisStreamServer:
    """Server for streaming Tetris frames over HTTP"""
    def __init__(self, port=8080, worker_id=0):
        self.port = port
        self.worker_id = worker_id
        self.server = None
        self.server_thread = None
        self.running = False
    
    def start(self):
        """Start the streaming server in a background thread"""
        if self.server_thread is not None and self.server_thread.is_alive():
            return
        
        self.running = True
        
        # Find an available port
        port = self.port
        max_port = self.port + 100  # Try up to 100 ports
        
        while port < max_port:
            try:
                self.server = ThreadedHTTPServer(('localhost', port), TetrisStreamHandler, worker_id=self.worker_id)
                print(f"Streaming server for worker {self.worker_id} started on http://localhost:{port}")
                break
            except socket.error:
                print(f"Port {port} is in use, trying next port...")
                port += 1
        
        if self.server is None:
            print(f"Could not find an available port for streaming server (worker {self.worker_id})")
            self.running = False
            return
        
        # Start server in a separate thread
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
    
    def update_frame(self, frame):
        """Update the current frame being streamed"""
        if self.server and self.running:
            # Convert frame to BGR if grayscale
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            self.server.update_frame(frame)
    
    def stop(self):
        """Stop the streaming server"""
        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
            self.server_thread = None
        
        self.server = None
        print(f"Streaming server for worker {self.worker_id} stopped")

class TrainingStatsHandler(BaseHTTPRequestHandler):
    """HTTP Handler for displaying training statistics dashboard"""
    
    def do_GET(self):
        if self.path == '/':
            # Serve HTML dashboard page
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Get worker URLs
            worker_links = ""
            for worker_id, port in self.server.worker_ports.items():
                worker_links += f'<a href="http://localhost:{port}" target="_blank" class="worker-link">Worker {worker_id}</a>'
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Tetris Training Dashboard</title>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .card {{ background-color: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
                    .stat-box {{ background: #f9f9f9; border-radius: 5px; padding: 15px; text-align: center; }}
                    .stat-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; color: #2196F3; }}
                    .stat-label {{ font-size: 14px; color: #777; }}
                    .chart-container {{ height: 300px; margin-top: 20px; }}
                    .worker-links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }}
                    .worker-link {{ 
                        display: inline-block;
                        padding: 8px 15px;
                        background-color: #2196F3;
                        color: white;
                        text-decoration: none;
                        border-radius: 4px;
                        transition: background-color 0.2s;
                    }}
                    .worker-link:hover {{ background-color: #0b7dda; }}
                    .section-header {{ display: flex; justify-content: space-between; align-items: center; }}
                    .status-indicator {{ 
                        display: inline-block;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        margin-left: 8px;
                    }}
                    .status-active {{ background-color: #4CAF50; }}
                    .status-inactive {{ background-color: #F44336; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="card">
                        <div class="section-header">
                            <h1>Tetris Training Dashboard</h1>
                            <div>
                                Status: <span id="training-status">Loading...</span>
                                <span id="status-indicator" class="status-indicator status-active"></span>
                            </div>
                        </div>
                        
                        <div class="stats-grid">
                            <div class="stat-box">
                                <div class="stat-label">Episodes Completed</div>
                                <div id="episodes-completed" class="stat-value">0</div>
                                <div class="stat-label">of <span id="total-episodes">0</span></div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Average Reward</div>
                                <div id="avg-reward" class="stat-value">0.0</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Average Steps</div>
                                <div id="avg-steps" class="stat-value">0.0</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Current Cycle</div>
                                <div id="current-cycle" class="stat-value">0</div>
                                <div class="stat-label">of <span id="total-cycles">0</span></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Learning Progress</h2>
                        <div class="chart-container">
                            <canvas id="rewardsChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Training Metrics</h2>
                        <div class="chart-container">
                            <canvas id="metricsChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Active Workers</h2>
                        <div class="worker-links">
                            {worker_links}
                        </div>
                    </div>
                </div>
                
                <script>
                    // Initialize charts
                    const rewardsCtx = document.getElementById('rewardsChart').getContext('2d');
                    const rewardsChart = new Chart(rewardsCtx, {{
                        type: 'line',
                        data: {{
                            labels: [],
                            datasets: [{{
                                label: 'Average Reward',
                                borderColor: '#2196F3',
                                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                                borderWidth: 2,
                                data: [],
                                fill: true,
                                tension: 0.4
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            interaction: {{ intersect: false, mode: 'index' }},
                            scales: {{
                                y: {{ beginAtZero: false }},
                                x: {{ title: {{ display: true, text: 'Episodes' }} }}
                            }}
                        }}
                    }});
                    
                    const metricsCtx = document.getElementById('metricsChart').getContext('2d');
                    const metricsChart = new Chart(metricsCtx, {{
                        type: 'line',
                        data: {{
                            labels: [],
                            datasets: [
                                {{
                                    label: 'Policy Loss',
                                    borderColor: '#F44336',
                                    borderWidth: 2,
                                    data: [],
                                    fill: false,
                                    tension: 0.4,
                                    yAxisID: 'y'
                                }},
                                {{
                                    label: 'Value Loss',
                                    borderColor: '#4CAF50',
                                    borderWidth: 2,
                                    data: [],
                                    fill: false,
                                    tension: 0.4,
                                    yAxisID: 'y'
                                }},
                                {{
                                    label: 'Entropy',
                                    borderColor: '#FF9800',
                                    borderWidth: 2,
                                    data: [],
                                    fill: false,
                                    tension: 0.4,
                                    yAxisID: 'y1'
                                }}
                            ]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            interaction: {{ intersect: false, mode: 'index' }},
                            scales: {{
                                y: {{ 
                                    type: 'linear',
                                    display: true,
                                    position: 'left',
                                    title: {{ display: true, text: 'Loss Values' }}
                                }},
                                y1: {{
                                    type: 'linear',
                                    display: true,
                                    position: 'right',
                                    title: {{ display: true, text: 'Entropy' }},
                                    grid: {{ drawOnChartArea: false }}
                                }},
                                x: {{ title: {{ display: true, text: 'Episodes' }} }}
                            }}
                        }}
                    }});
                    
                    // Function to update dashboard data
                    function updateDashboard() {{
                        fetch('/stats')
                            .then(response => response.json())
                            .then(data => {{
                                // Update basic stats
                                document.getElementById('episodes-completed').textContent = data.episodes_completed;
                                document.getElementById('total-episodes').textContent = data.total_episodes;
                                document.getElementById('avg-reward').textContent = data.avg_reward.toFixed(2);
                                document.getElementById('avg-steps').textContent = data.avg_steps.toFixed(1);
                                document.getElementById('current-cycle').textContent = data.current_cycle;
                                document.getElementById('total-cycles').textContent = data.total_cycles;
                                
                                // Update status
                                document.getElementById('training-status').textContent = data.status;
                                const statusIndicator = document.getElementById('status-indicator');
                                statusIndicator.className = "status-indicator " + 
                                    (data.status === "Running" ? "status-active" : "status-inactive");
                                
                                // Update rewards chart
                                if (data.reward_x && data.reward_y) {{
                                    rewardsChart.data.labels = data.reward_x;
                                    rewardsChart.data.datasets[0].data = data.reward_y;
                                    rewardsChart.update();
                                }}
                                
                                // Update metrics chart
                                if (data.metrics_x && data.metrics_y) {{
                                    metricsChart.data.labels = data.metrics_x;
                                    
                                    // Policy loss
                                    if (data.metrics_y.policy) {{
                                        metricsChart.data.datasets[0].data = data.metrics_y.policy;
                                    }}
                                    
                                    // Value loss
                                    if (data.metrics_y.value) {{
                                        metricsChart.data.datasets[1].data = data.metrics_y.value;
                                    }}
                                    
                                    // Entropy
                                    if (data.metrics_y.entropy) {{
                                        metricsChart.data.datasets[2].data = data.metrics_y.entropy;
                                    }}
                                    
                                    metricsChart.update();
                                }}
                            }})
                            .catch(err => {{
                                console.error("Error updating dashboard:", err);
                                document.getElementById('training-status').textContent = "Connection Error";
                                document.getElementById('status-indicator').className = "status-indicator status-inactive";
                            }});
                    }}
                    
                    // Initial update and set interval
                    updateDashboard();
                    setInterval(updateDashboard, 2000);
                </script>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode())
            
        elif self.path == '/stats':
            # Serve training stats as JSON
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.end_headers()
            
            # Get current training stats
            stats = self.server.get_current_stats()
            self.wfile.write(json.dumps(stats).encode())
            
        elif self.path.startswith('/chart/'):
            # Generate chart images on demand
            chart_type = self.path[7:]
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            
            img_data = self.server.generate_chart(chart_type)
            self.wfile.write(img_data)
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Disable logging to prevent console spam
        return

class TrainingStatsServer(ThreadingMixIn, HTTPServer):
    """Server for displaying training statistics dashboard"""
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.stats = {
            'episodes_completed': 0,
            'total_episodes': 0,
            'avg_reward': 0.0,
            'avg_steps': 0.0,
            'current_cycle': 0,
            'total_cycles': 0,
            'status': 'Initializing',
            'reward_x': [],
            'reward_y': [],
            'metrics_x': [],
            'metrics_y': {
                'policy': [],
                'value': [],
                'entropy': []
            }
        }
        self.worker_ports = {}  # Map of worker_id to port
        self.last_update = time.time()
    
    def update_stats(self, stats_dict: Dict[str, Any]):
        """Update training statistics"""
        self.stats.update(stats_dict)
        self.last_update = time.time()
        print(f"Dashboard stats updated: {stats_dict.get('status')} - Episodes: {stats_dict.get('episodes_completed', 0)}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current training statistics"""
        # If last update was too long ago, consider training inactive
        if time.time() - self.last_update > 120:  # Increased from 30 to 120 seconds
            self.stats['status'] = 'Inactive'
        return self.stats
    
    def add_worker(self, worker_id: int, port: int):
        """Add a worker to the dashboard"""
        self.worker_ports[worker_id] = port
    
    def generate_chart(self, chart_type: str) -> bytes:
        """Generate chart image based on current stats"""
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'rewards':
            plt.plot(self.stats['reward_x'], self.stats['reward_y'])
            plt.title('Training Rewards')
            plt.xlabel('Episodes')
            plt.ylabel('Average Reward')
        elif chart_type == 'metrics':
            if self.stats['metrics_x']:
                plt.plot(self.stats['metrics_x'], self.stats['metrics_y']['policy'], label='Policy Loss')
                plt.plot(self.stats['metrics_x'], self.stats['metrics_y']['value'], label='Value Loss')
                plt.title('Training Metrics')
                plt.xlabel('Episodes')
                plt.ylabel('Loss')
                plt.legend()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()

class TrainingDashboard:
    """Dashboard for displaying training statistics"""
    def __init__(self, base_port=8080):
        self.port = base_port
        self.server = None
        self.server_thread = None
        self.running = False
        self.worker_servers = {}
    
    def start(self):
        """Start the dashboard server"""
        if self.server_thread is not None and self.server_thread.is_alive():
            return
        
        self.running = True
        
        # Find an available port
        port = self.port
        max_port = self.port + 10  # Try up to 10 ports
        
        while port < max_port:
            try:
                self.server = TrainingStatsServer(('localhost', port), TrainingStatsHandler)
                print(f"Training dashboard started on http://localhost:{port}")
                break
            except socket.error:
                print(f"Port {port} is in use, trying next port...")
                port += 1
        
        if self.server is None:
            print(f"Could not find an available port for training dashboard")
            self.running = False
            return
        
        # Start server in a separate thread
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        
        # Set initial stats immediately
        self.update_stats({
            'episodes_completed': 0,
            'total_episodes': 1000,  # Default value, will be updated later
            'avg_reward': 0.0,
            'avg_steps': 0.0,
            'current_cycle': 0,
            'total_cycles': 1,
            'status': 'Starting',
            'reward_x': [0],
            'reward_y': [0],
        })
    
    def update_stats(self, stats_dict: Dict[str, Any]):
        """Update training statistics"""
        if self.server and self.running:
            self.server.update_stats(stats_dict)
    
    def add_worker(self, worker_id: int, port: int):
        """Register a worker with the dashboard"""
        if self.server and self.running:
            self.server.add_worker(worker_id, port)
    
    def stop(self):
        """Stop the dashboard server"""
        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
            self.server_thread = None
        
        self.server = None
