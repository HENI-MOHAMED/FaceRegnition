import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import json
import numpy as np
from imgbeddings import imgbeddings
import math
from time import sleep
import datetime
import sys

class FolderProtectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Folder Face Protection")
        self.root.geometry("400x600")
        
        # Variables
        self.folder_path = tk.StringVar()
        self.photo_path = tk.StringVar()
        self.is_protecting = False
        self.config_file = "folder_protector_config.json"
        self.startup_var = tk.BooleanVar(value=False)
        
        # Check if already in startup
        startup_path = os.path.join(os.getenv('APPDATA'), 
                                  r'Microsoft\Windows\Start Menu\Programs\Startup\FolderProtector.bat')
        
        # Initialize face detection with proper paths
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            haar_cascade_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
            model_path = os.path.join(script_dir, "patch32_v1.onnx")
            
            if not os.path.exists(haar_cascade_path):
                messagebox.showerror("Error", "Could not find haarcascade_frontalface_default.xml")
                sys.exit(1)
            if not os.path.exists(model_path):
                messagebox.showerror("Error", "Could not find patch32_v1.onnx")
                sys.exit(1)
                
            self.haar_cascade = cv2.CascadeClassifier(haar_cascade_path)
            self.ibed = imgbeddings(model_path=model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize face detection: {str(e)}")
            sys.exit(1)
            
        # Create necessary directories
        if not os.path.exists("stored-faces"):
            os.makedirs("stored-faces")
            
        # Load saved settings
        self.load_saved_settings()
        
        # Create GUI elements
        self.create_gui()
            
    def create_gui(self):
        # Main title
        tk.Label(self.root, text="Folder Face Protection", font=('Helvetica', 16, 'bold')).pack(pady=20)
        
        # Folder selection
        folder_frame = tk.Frame(self.root)
        folder_frame.pack(fill='x', padx=20, pady=5)
        tk.Label(folder_frame, text="Select folder to protect:").pack(anchor='w')
        tk.Entry(folder_frame, textvariable=self.folder_path, width=40).pack(side='left', pady=5)
        tk.Button(folder_frame, text="Browse", command=self.browse_folder).pack(side='right', padx=5)
        
        # Photo selection
        photo_frame = tk.Frame(self.root)
        photo_frame.pack(fill='x', padx=20, pady=5)
        tk.Label(photo_frame, text="Select your photo:").pack(anchor='w')
        tk.Entry(photo_frame, textvariable=self.photo_path, width=40).pack(side='left', pady=5)
        tk.Button(photo_frame, text="Browse", command=self.browse_photo).pack(side='right', padx=5)
        
        # Photo preview
        preview_frame = tk.Frame(self.root)
        preview_frame.pack(pady=10)
        self.preview_label = tk.Label(preview_frame, text="No photo selected")
        self.preview_label.pack()
        
        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill='x', padx=20, pady=10)
        
        self.start_button = tk.Button(button_frame, text="Start Protection", 
                                    command=self.start_protection, 
                                    bg="#4CAF50", fg="white", height=2)
        self.start_button.pack(fill='x', pady=5)
        
        self.stop_button = tk.Button(button_frame, text="Stop Protection", 
                                   command=self.stop_protection, 
                                   bg="#f44336", fg="white", height=2, 
                                   state='disabled')
        self.stop_button.pack(fill='x', pady=5)
        
        # Status section
        status_frame = tk.Frame(self.root, relief='sunken', borderwidth=1)
        status_frame.pack(fill='x', padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, text="Status: Ready", 
                                   font=('Helvetica', 10))
        self.status_label.pack(pady=10)
        
        # Minimize to tray checkbox
        self.minimize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Minimize to tray when running", 
                      variable=self.minimize_var).pack(pady=5)
                      
        # Run at startup checkbox
        startup_path = os.path.join(os.getenv('APPDATA'), 
                                  r'Microsoft\Windows\Start Menu\Programs\Startup\FolderProtector.bat')
        self.startup_var.set(os.path.exists(startup_path))
        tk.Checkbutton(self.root, text="Run at startup", 
                      variable=self.startup_var,
                      command=self.toggle_startup).pack(pady=5)
    
    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)
            self.save_settings()
    
    def browse_photo(self):
        photo = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        if photo:
            self.photo_path.set(photo)
            self.update_preview(photo)
            if self.process_user_photo(photo):
                self.save_settings()
    
    def update_preview(self, photo_path):
        try:
            # Open and resize image for preview
            image = Image.open(photo_path)
            image.thumbnail((150, 150))
            photo = ImageTk.PhotoImage(image)
            self.preview_label.config(image=photo)
            self.preview_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Could not load preview: {str(e)}")
    
    def process_user_photo(self, photo_path):
        """Process the user's photo and store face embedding"""
        try:
            # Read and process the image
            img = cv2.imread(photo_path, 0)
            if img is None:
                raise Exception("Could not read the image")

            # Detect faces
            gray_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            faces = self.haar_cascade.detectMultiScale(
                gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(110, 110)
            )

            if len(faces) == 0:
                raise Exception("No face detected in the photo")

            if len(faces) > 1:
                raise Exception("Multiple faces detected. Please use a photo with only your face")

            # Get the face and calculate embedding
            x, y, w, h = faces[0]
            face_img = img[y:y + h, x:x + w]
            cv2.imwrite('stored-faces/user.jpg', face_img)
            
            # Calculate and store embedding
            pil_img = Image.fromarray(face_img)
            embedding = self.ibed.to_embeddings(pil_img)
            
            # Save embedding
            with open("face_embeddings.json", 'w') as f:
                json.dump({"user.jpg": embedding[0].tolist()}, f)
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False

    def verify_face(self):
        """Verify face using webcam"""
        try:
            # Load stored embedding
            with open("face_embeddings.json", 'r') as f:
                stored_embedding = json.load(f)["user.jpg"]

            # Initialize webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not access webcam")

            attempt_count = 0
            while attempt_count < 100:  # Limit verification attempts
                ret, frame = cap.read()
                if not ret:
                    continue

                # Process frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=2, minSize=(110, 110)
                )

                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = gray[y:y + h, x:x + w]
                    pil_img = Image.fromarray(face_img)
                    
                    # Calculate embedding and compare
                    embedding = self.ibed.to_embeddings(pil_img)
                    similarity = self.cosine_similarity(stored_embedding, embedding[0].tolist())
                    
                    if similarity > 0.94:
                        cap.release()
                        return True

                attempt_count += 1
                
            cap.release()
            return False
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False

    @staticmethod
    def cosine_similarity(a, b):
        dot_product = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai ** 2 for ai in a))
        norm_b = math.sqrt(sum(bi ** 2 for bi in b))
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

    def start_protection(self):
        if not self.folder_path.get() or not self.photo_path.get():
            messagebox.showerror("Error", "Please select both folder and photo")
            return
            
        if not self.process_user_photo(self.photo_path.get()):
            return
            
        self.is_protecting = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_label.config(text="Status: Protection Active")
        
        if self.minimize_var.get():
            self.root.iconify()
            
        self.start_monitoring()
        
    def stop_protection(self):
        self.is_protecting = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Status: Protection Stopped")
    
    def save_settings(self):
        """Save current settings to config file"""
        settings = {
            "folder_path": self.folder_path.get(),
            "photo_path": self.photo_path.get()
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
    
    def toggle_startup(self):
        """Toggle application startup with Windows"""
        startup_path = os.path.join(os.getenv('APPDATA'), 
                                  r'Microsoft\Windows\Start Menu\Programs\Startup\FolderProtector.bat')
        
        if self.startup_var.get():
            try:
                # Create batch file content
                script_path = os.path.abspath(sys.argv[0])
                batch_content = f'@echo off\nstart "" pythonw "{script_path}" --autostart'
                
                # Write the batch file
                with open(startup_path, 'w') as f:
                    f.write(batch_content)
                    
                messagebox.showinfo("Startup", "Application will now run at startup")
            except Exception as e:
                messagebox.showerror("Error", f"Could not set up startup: {str(e)}")
                self.startup_var.set(False)
        else:
            try:
                if os.path.exists(startup_path):
                    os.remove(startup_path)
                messagebox.showinfo("Startup", "Application will no longer run at startup")
            except Exception as e:
                messagebox.showerror("Error", f"Could not remove startup: {str(e)}")
                self.startup_var.set(True)

    def load_saved_settings(self):
        """Load saved settings from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    settings = json.load(f)
                
                # Restore folder path
                if os.path.exists(settings.get("folder_path", "")):
                    self.folder_path.set(settings["folder_path"])
                
                # Restore photo path and face data
                photo_path = settings.get("photo_path", "")
                if os.path.exists(photo_path):
                    self.photo_path.set(photo_path)
                    # Process the photo again to ensure face data is current
                    self.process_user_photo(photo_path)
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_protecting:
            if messagebox.askokcancel("Quit", 
                "Protection is active. Do you want to stop protection and quit?"):
                self.stop_protection()
                self.root.quit()
        else:
            self.root.quit()

    def start_monitoring(self):
        """Monitor the protected folder for access attempts"""
        folder_path = self.folder_path.get()
        s = os.stat(folder_path)
        last_access_time = datetime.datetime.fromtimestamp(s.st_atime)
        
        def check_folder():
            if not self.is_protecting:
                return
                
            nonlocal last_access_time
            try:
                stats = os.stat(folder_path)
                current_access = datetime.datetime.fromtimestamp(stats.st_atime)
                
                if current_access > last_access_time:
                    self.status_label.config(text="Status: Verifying Identity...")
                    self.root.update()
                    
                    if not self.verify_face():
                        # Close the folder if unauthorized
                        try:
                            
                            # Additional measure: close the specific folder
                            os.system(f'taskkill /F /IM explorer.exe')
                            os.system('start explorer.exe')
                            
                            messagebox.showwarning("Security Alert", 
                                                 "Unauthorized access attempted - Folder has been closed")
                        except Exception as e:
                            print(f"Error closing folder: {str(e)}")
                    
                    last_access_time = current_access
                    
            except FileNotFoundError:
                messagebox.showerror("Error", f"The folder '{folder_path}' was not found.")
                self.stop_protection()
                return
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                self.stop_protection()
                return
            
            # Schedule next check
            self.root.after(500, check_folder)  # Check every half second
        
        # Start the monitoring loop
        check_folder()


if __name__ == "__main__":
    # Check if running in autostart mode
    autostart = "--autostart" in sys.argv
    
    root = tk.Tk()
    app = FolderProtectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # Handle window closing
    
    # If in autostart mode and we have saved settings, start protection automatically
    if autostart and os.path.exists(app.config_file):
        with open(app.config_file, 'r') as f:
            settings = json.load(f)
            if all(os.path.exists(settings.get(k, "")) for k in ["folder_path", "photo_path"]):
                root.after(1000, app.start_protection)  # Start protection after 1 second
                root.iconify()  # Minimize window
    
    root.mainloop()


