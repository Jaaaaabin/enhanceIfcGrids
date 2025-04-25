import os
import time
import numpy as np
import webbrowser
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

class SolutionFinderApp(tk.Tk):
    
    def __init__(self, base_directory):
        super().__init__()
        self.base_directory = base_directory
        self.title("Reference Grid Solution Finder")

        # Set the window to 95% of the full screen size
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = int(screen_width * 0.95)
        window_height = int(screen_height * 0.85)
        self.geometry(f"{window_width}x{window_height}")  # Set the size to 95% of screen size

        self.configure(bg='#ffffff')  # Set background color

        # Create input fields and labels in a frame
        self.create_widgets()

    def create_widgets(self):
        # Main frame to hold everything
        main_frame = tk.Frame(self, padx=20, pady=20, bg='#ffffff')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title label (full row span)
        header_label = tk.Label(
            main_frame,
            text="Reference Grid Solution Finder",
            font=("Helvetica", 16), bg='#ffffff')
        header_label.grid(row=0, column=1, columnspan=3, pady=(0, 20), sticky="we")

        # Left-side form (labels, entry, radio buttons)
        form_frame = tk.Frame(main_frame, bg='#ffffff')
        form_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nw")

        # Case Number
        tk.Label(form_frame, text="Case Number:", font=("Helvetica", 10), bg='#ffffff').grid(row=1, column=0, sticky="e")
        self.case_number = tk.Entry(form_frame)
        self.case_number.grid(row=1, column=1, pady=5)

        # Category: Radio buttons
        tk.Label(form_frame, text="Category:", font=("Helvetica", 10), bg='#ffffff').grid(row=2, column=0, sticky="e")
        self.category_var = tk.StringVar(value="pft")  # Default to 'pft'
        self.pft_radio = tk.Radiobutton(form_frame, text="pareto_front", variable=self.category_var, value="pft", bg='#ffffff')
        self.non_radio = tk.Radiobutton(form_frame, text="non_pareto_front", variable=self.category_var, value="non", bg='#ffffff')
        self.pft_radio.grid(row=3, column=1, sticky="w", pady=5)
        self.non_radio.grid(row=4, column=1, sticky="w", pady=5)

        # Submit button to find the solution
        self.submit_button = tk.Button(
            form_frame,
            text="Find Solution",
            command=self.find_solution,
            bg='#d9d9d9',
            font=("Helvetica", 12))
        self.submit_button.grid(
            row=5,
            column=1,
            columnspan=3,
            pady=20)

        # Exit button
        self.exit_button = tk.Button(
            form_frame,
            text="Exit",
            command=self.quit,
            bg='#d9d9d9',
            font=("Helvetica", 12))
        self.exit_button.grid(
            row=6,
            column=1,
            columnspan=3,
            pady=20)

        # Frame for the image
        self.image_frame = tk.Frame(main_frame, bg='#ffffff')
        self.image_frame.grid(row=1, column=3, padx=20, pady=20, sticky="n")

        # Target Y slider (between Category region and Image)
        tk.Label(main_frame, text="f_distribution", font=("Helvetica", 10), bg='#ffffff').grid(
            row=1, column=2, sticky="n", padx=(10, 10))
        self.target_y_var = tk.DoubleVar(value=0.0)
        self.target_y_slider = tk.Scale(
            main_frame, from_=1.0, to=0.0, orient=tk.VERTICAL, variable=self.target_y_var,
            resolution=0.001, bg='#ffffff', fg='black', length=200)
        self.target_y_slider.grid(row=1, column=2, sticky="ns", pady=20)

        # Target X slider (below the image)
        tk.Label(self.image_frame, text="f_unbound", font=("Helvetica", 10), bg='#ffffff').grid(
            row=2, column=0, sticky="w")
        self.target_x_var = tk.DoubleVar(value=0.0)
        self.target_x_slider = tk.Scale(self.image_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.target_x_var, resolution=0.001, bg='#ffffff', fg='black')
        self.target_x_slider.grid(row=3, column=0, pady=5, sticky="ew")
        
        # Load and display the static PNG image on the right side
        self.load_image(self.image_frame)

    def load_image(self, parent_frame, image_subfolder_name='res_ga'):
        """
        Loads a PNG image and displays it on the right side of the GUI.
        """
        image_path = os.path.join(self.base_directory, image_subfolder_name, "all_rr_true.png")  # Provide the correct path to your image
        try:
            img = Image.open(image_path)
            img = img.resize((450, 450), Image.Resampling.LANCZOS)  # Resize image to fit nicely in the GUI
            self.photo = ImageTk.PhotoImage(img)  # Keep a reference to avoid garbage collection

            # Display the image in a label
            img_label = tk.Label(parent_frame, image=self.photo, bg='#ffffff')
            img_label.grid(row=0, column=0, padx=20, pady=20, sticky="nw")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def find_solution(self):
        case_number = self.case_number.get()
        category = self.category_var.get()
        target_x = float(self.target_x_var.get())
        target_y = float(self.target_y_var.get())

        # Validate category
        if category not in ['pft', 'non']:
            messagebox.showerror("Error", "Category must be 'pft' for pareto_front or 'non' for non_pareto_front.")
            return

        # Initialize the SolutionFinder and find the solution
        try:
            finder = SolutionFinder(self.base_directory)
            finder.find_closest_solution(case_number, category, target_x, target_y)
            messagebox.showinfo("Success", "Solutions found and HTML files opened.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

class SolutionFinder:
    
    def __init__(self, base_directory):
        self.base_directory = base_directory

    def find_closest_solution(self, case_number, category, target_x, target_y):
        subfolder_path = self.get_closest_subfolder_path(case_number, category)
        if not os.path.exists(subfolder_path):
            raise FileNotFoundError(f"Specified folder '{subfolder_path}' does not exist!")
        closest_folder = self.load_solutions_from_folder(subfolder_path, target_x, target_y)
        self.open_html_files(closest_folder)

    def get_closest_subfolder_path(self, case_number, category, solution_subfolder_name='solution_ga'):
        base_folder = os.path.join(self.base_directory, solution_subfolder_name)
        all_subfolder_cases = [f.name for f in os.scandir(base_folder) if f.is_dir()]

        closest_subfolder = None
        min_distance = float('inf')

        for f in all_subfolder_cases:
            try:
                folder_number = int(f.split('-')[0])
                distance = abs(folder_number - int(case_number))
                if distance < min_distance:
                    min_distance = distance
                    closest_subfolder = f
            except ValueError:
                continue

        if closest_subfolder is None:
            raise FileNotFoundError(f"No matching subfolders found for case_number '{case_number}'")

        subfolder_category = 'pareto_front' if category == 'pft' else 'non_pareto_front'
        return os.path.join(base_folder, closest_subfolder, subfolder_category)

    def load_solutions_from_folder(self, folder_path, target_x, target_y):
        closest_folder = None
        min_distance = float('inf')

        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)

            if not os.path.isdir(subfolder_path):
                continue

            try:
                subfolder_x, subfolder_y = map(float, subfolder_name.strip('()').split(','))
            except ValueError:
                continue

            distance = np.sqrt((target_x - subfolder_x) ** 2 + (target_y - subfolder_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_folder = subfolder_path

        if closest_folder is None:
            raise ValueError(f"No valid subfolders found in folder '{folder_path}' with proper (x, y) format.")

        print(f"Closest folder found for the input (x:{target_x}, y:{target_y})")
        print(f"{closest_folder}")
    
        return closest_folder

    def open_html_files(self, folder_path):
        html_files = [f for f in os.listdir(folder_path) if f.endswith('merging.html')]

        if not html_files:
            print(f"No HTML files found in folder: {folder_path}")
            return

        for file_name in html_files:
            file_path = os.path.join(folder_path, file_name)
            print(f"Opening {file_name}...")
            webbrowser.open(f'file://{os.path.realpath(file_path)}')
            time.sleep(0.5)

if __name__ == "__main__":
    base_directory = os.getcwd()
    app = SolutionFinderApp(base_directory)
    app.mainloop()



# Save.
# if __name__ == "__main__":

#     # Check if correct number of arguments is provided
#     if len(sys.argv) != 5:
#         print("Usage: python find_solution.py <case_number> <category: pft/non> <target_x> <target_y>")
#         sys.exit(1)

#     # Get input parameters from the command line
#     case_number = sys.argv[1]
#     category = sys.argv[2]
#     target_x = float(sys.argv[3])
#     target_y = float(sys.argv[4])

#     # Validate category
#     if category not in ['pft', 'non']:
#         print("Error: category must be 'pft' for pareto_front or 'non' for non_pareto_front.")
#         sys.exit(1)

#     # Define base directory as the current working directory
#     base_directory = os.getcwd()

#     # Initialize the SolutionFinder
#     finder = SolutionFinder(base_directory)

#     # Find the closest subfolder and open all HTML files
#     try:
#         finder.find_closest_solution(case_number, category, target_x, target_y)
#     except Exception as e:
#         print(f"Error: {e}")
