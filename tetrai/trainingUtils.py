import glfw
import numpy as np
import threading
from multiprocessing import Queue
import queue
import cv2
import moderngl
import numpy as np
import torch

class DisplayManager:
    def __init__(self, width=512, height=512, title="Tetris Training"):
        self.width = width
        self.height = height
        self.title = title
        self.queue = Queue()
        self.running = True
        self.display_thread = None

        # OpenGL/GLFW initialization will happen in the display thread
        self.ctx = None
        self.prog = None
        self.vbo = None
        self.texture = None
        
    def start(self):
        self.display_thread = threading.Thread(target=self._run_display)
        self.display_thread.start()
        
    def stop(self):
        self.running = False
        if self.queue:
            self.queue.put(None)  # Signal to stop
        if self.display_thread:
            self.display_thread.join()

    def _init_gl(self):
        # Initialize GLFW
        if not glfw.init():
            return False

        # Configure GLFW
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)

        # Create window
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            return False

        glfw.make_context_current(self.window)
        
        # Create ModernGL context
        self.ctx = moderngl.create_context()
        
        # Create shader program
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    v_texcoord = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D texture0;
                in vec2 v_texcoord;
                out vec4 f_color;
                void main() {
                    f_color = texture(texture0, v_texcoord);
                }
            '''
        )

        # Create vertex buffer
        vertices = np.array([
            # positions  # texture coords
            -1.0, -1.0,  0.0, 1.0,
            1.0, -1.0,   1.0, 1.0,
            1.0,  1.0,   1.0, 0.0,
            -1.0,  1.0,  0.0, 0.0,
        ], dtype='f4')

        indices = np.array([0, 1, 2, 0, 2, 3], dtype='i4')
        
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '2f 2f', 'in_position', 'in_texcoord'),
            ],
            self.ibo
        )

        # Create texture
        self.texture = self.ctx.texture((self.width, self.height), 3)
        self.texture.use(0)
        
        return True

    def _run_display(self):
        if not self._init_gl():
            print("Failed to initialize OpenGL")
            return

        while self.running and not glfw.window_should_close(self.window):
            try:
                frame = self.queue.get(timeout=0.1)
                if frame is None:
                    break
                    
                # Convert frame to RGB if necessary
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                elif frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Update texture
                self.texture.write(frame.tobytes())
                
                # Clear and render
                self.ctx.clear(0.0, 0.0, 0.0)
                self.vao.render()
                
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Display error: {e}")
                break

        # Cleanup
        self.ctx.release()
        glfw.destroy_window(self.window)
        glfw.terminate()

    def update(self, frame: torch.Tensor):
        """Update display with torch tensor from TetrisEnv
        
        Args:
            frame: torch tensor of shape (H,W,3) in RGB format
        """
        if not self.running:
            return
            
        try:
            # Clear queue if too full to prevent lag
            while self.queue.qsize() > 2:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
                    
            # Convert to numpy only at the last moment for OpenGL
            if frame.is_cuda:
                frame = frame.cpu()
            
            # Ensure we're working with uint8
            if frame.dtype != torch.uint8:
                frame = (frame * 255).to(torch.uint8)
                
            # Handle different tensor shapes
            if len(frame.shape) == 2:
                frame = frame.unsqueeze(-1).repeat(1, 1, 3)
            elif frame.shape[-1] == 4:
                frame = frame[...,:3]
                
            self.queue.put(frame.numpy())
            
        except Exception as e:
            print(f"Error in display update: {e}")