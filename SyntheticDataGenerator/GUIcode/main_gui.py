import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import yaml


class DataConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Configuration Tool")
        self.root.geometry("900x700")

        # State variables
        self.config_data = {}
        self.selected_states = []
        self.state_checkbuttons = {}

        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.start_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.start_tab, text="Start Tab")

        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="Specifying the first model")

        self.setup_start_tab()
        self.setup_model_tab()

    def setup_start_tab(self):
        """Setup the Start Tab with all configuration options."""
        main_frame = ttk.Frame(self.start_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Load configuration file
        config_frame = ttk.LabelFrame(scrollable_frame, text="Load Configuration File", padding=5)
        config_frame.pack(fill=tk.X, pady=5)

        ttk.Button(config_frame, text="Load Configuration", command=self.load_config).pack(side=tk.LEFT, padx=5)
        self.config_label = ttk.Label(config_frame, text="No config loaded")
        self.config_label.pack(side=tk.LEFT, padx=5)

        # States selection
        states_frame = ttk.LabelFrame(scrollable_frame, text="Select States", padding=5)
        states_frame.pack(fill=tk.X, pady=5)

        all_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                  "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                  "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                  "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

        ttk.Button(states_frame, text="Select All", command=lambda: self.select_all_states(all_states)).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(states_frame, text="Deselect All", command=self.deselect_all_states).pack(side=tk.LEFT, padx=5)

        states_check_frame = ttk.Frame(states_frame)
        states_check_frame.pack(fill=tk.X, padx=5, pady=5)

        for i, state in enumerate(all_states):
            var = tk.BooleanVar()
            self.state_checkbuttons[state] = var
            ttk.Checkbutton(states_check_frame, text=state, variable=var).grid(row=i//10,column=i%10, padx=3)

        # Year and Quarter
        params_frame = ttk.LabelFrame(scrollable_frame, text="Year and Quarter", padding=5)
        params_frame.pack(fill=tk.X, pady=5)

        ttk.Label(params_frame, text="Year (1998-2016):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.year_var = tk.StringVar(value="2016")
        ttk.Spinbox(params_frame, from_=1998, to=2016, textvariable=self.year_var, width=10).grid(row=0, column=1,
                                                                                                  sticky=tk.W, padx=5)

        ttk.Label(params_frame, text="Quarter (1-4):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.qtr_var = tk.StringVar(value="1")
        ttk.Spinbox(params_frame, from_=1, to=4, textvariable=self.qtr_var, width=10).grid(row=1, column=1, sticky=tk.W,
                                                                                           padx=5)

        # Random Seed and Thresholds
        seed_frame = ttk.LabelFrame(scrollable_frame, text="Model Parameters", padding=5)
        seed_frame.pack(fill=tk.X, pady=5)

        ttk.Label(seed_frame, text="Random Seed:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.rseed_var = tk.StringVar(value="1")
        ttk.Entry(seed_frame, textvariable=self.rseed_var, width=15).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(seed_frame, text="Month 2 Noise Coefficient:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.m2emp_var = tk.StringVar(value="1.69")
        ttk.Entry(seed_frame, textvariable=self.m2emp_var, width=15).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(seed_frame, text="Cook's Distance Outlier Threshold:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.cooks_var = tk.StringVar(value="1")
        ttk.Entry(seed_frame, textvariable=self.cooks_var, width=15).grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(seed_frame, text="Studentized Residual Outlier Threshold:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.outlier_var = tk.StringVar(value="4.3")
        ttk.Entry(seed_frame, textvariable=self.outlier_var, width=15).grid(row=3, column=1, sticky=tk.W, padx=5)

        # Stage selection
        stage_frame = ttk.LabelFrame(scrollable_frame, text="Processing Stage", padding=5)
        stage_frame.pack(fill=tk.X, pady=5)

        self.stage_var = tk.StringVar(value="0")
        stages = [
            ("Stage 0: Download raw data", "0"),
            ("Stage 1: Combine raw data", "1"),
            ("Stage 2: Create complete county by NAICS-6 data", "2"),
            ("Stage 3: Disaggregate county by NAICS-6 data to establishment-level", "3")
        ]

        for text, value in stages:
            ttk.Radiobutton(stage_frame, text=text, variable=self.stage_var, value=value,
                            command=self.on_stage_change).pack(anchor=tk.W, padx=5, pady=3)

        # Census API Key (shown for Stage 0)
        self.api_frame = ttk.LabelFrame(scrollable_frame, text="Census API Key", padding=5)
        self.api_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.api_var = tk.StringVar(value="")
        ttk.Entry(self.api_frame, textvariable=self.api_var, width=30).grid(row=0, column=1, sticky=tk.W, padx=5)

        # Microdata parameters (shown for Stage 3)
        self.microdata_frame = ttk.LabelFrame(scrollable_frame, text="Microdata Parameters", padding=5)

        ttk.Label(self.microdata_frame, text="Prior Gamma Shape:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.gam_shape_var = tk.StringVar(value="3")
        ttk.Entry(self.microdata_frame, textvariable=self.gam_shape_var, width=15).grid(row=0, column=1, sticky=tk.W,
                                                                                        padx=5)

        ttk.Label(self.microdata_frame, text="Prior Gamma Scale:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.gam_scale_var = tk.StringVar(value="20")
        ttk.Entry(self.microdata_frame, textvariable=self.gam_scale_var, width=15).grid(row=1, column=1, sticky=tk.W,
                                                                                        padx=5)

        # Data folders
        folder_frame = ttk.LabelFrame(scrollable_frame, text="Data Folder Locations", padding=5)
        folder_frame.pack(fill=tk.X, pady=5)

        ttk.Label(folder_frame, text="Data Folder:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.data_folder_var = tk.StringVar(value="DataDiag/")
        ttk.Entry(folder_frame, textvariable=self.data_folder_var, width=40).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(folder_frame, text="Browse", command=lambda: self.browse_folder(self.data_folder_var)).grid(row=0,
                                                                                                               column=2,
                                                                                                               padx=5)

        # File paths
        paths_frame = ttk.LabelFrame(scrollable_frame, text="File Paths", padding=5)
        paths_frame.pack(fill=tk.X, pady=5)

        paths = [
            ("Raw CBP Data File", "cbp_file"),
            ("Raw QWI Data Folder", "qwi_folder"),
            ("Raw QCEW Data Folder", "qcew_folder"),
            ("Imputed CBP File", "impute_cbp_file"),
            ("Combined Data File", "combined_file"),
            ("NAICS-6 File", "naics6_file"),
            ("Subset Establishment Folder", "subset_folder"),
            ("By State Establishment Folder", "state_folder")
        ]

        self.path_vars = {}
        for idx, (label, key) in enumerate(paths):
            ttk.Label(paths_frame, text=label + ":").grid(row=idx, column=0, sticky=tk.W, padx=5, pady=3)
            var = tk.StringVar(value="")
            self.path_vars[key] = var
            ttk.Entry(paths_frame, textvariable=var, width=40).grid(row=idx, column=1, sticky=tk.W, padx=5)
            ttk.Button(paths_frame, text="Browse", command=lambda v=var: self.browse_path(v)).grid(row=idx, column=2,
                                                                                                   padx=5)

        # Config file save location
        config_save_frame = ttk.LabelFrame(scrollable_frame, text="Configuration Save", padding=5)
        config_save_frame.pack(fill=tk.X, pady=5)

        ttk.Label(config_save_frame, text="Config File Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.config_name_var = tk.StringVar(value="config/config_2016_1.yaml")
        ttk.Entry(config_save_frame, textvariable=self.config_name_var, width=40).grid(row=0, column=1, sticky=tk.W,
                                                                                       padx=5)

        # Action buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Edit and Save Configuration File",
                   command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Use All Default Settings",
                   command=self.use_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Customize Prediction Models",
                   command=self.customize_models).pack(side=tk.LEFT, padx=5)

        # Pack scrollbar and canvas
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.on_stage_change()

    def setup_model_tab(self):
        """Setup the Specifying the first model tab."""
        frame = ttk.Frame(self.model_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Model Configuration Options", font=("Arial", 14, "bold")).pack(pady=10)
        ttk.Label(frame, text="Placeholder for model specification interface").pack(pady=20)

    def load_config(self):
        """Load configuration from YAML file."""
        file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.config_data = yaml.safe_load(f)
                self.config_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.populate_from_config()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {e}")

    def populate_from_config(self):
        """Populate GUI fields from loaded config."""
        gen_config = self.config_data.get('generalConfig', {})

        self.year_var.set(str(gen_config.get('YEAR', '2016')))
        self.qtr_var.set(str(gen_config.get('QTR', '1')))
        self.api_var.set(gen_config.get('API_KEY', ''))

        # Set states
        states = gen_config.get('STATES', [])
        for state, var in self.state_checkbuttons.items():
            var.set(state in states)

        pre_config = self.config_data.get('preprocessConfig', {})
        self.data_folder_var.set(pre_config.get('DATA_IN_FOLDER', 'DataDiag/'))

    def select_all_states(self, states):
        """Select all states."""
        for state in states:
            if state in self.state_checkbuttons:
                self.state_checkbuttons[state].set(True)

    def deselect_all_states(self):
        """Deselect all states."""
        for var in self.state_checkbuttons.values():
            var.set(False)

    def on_stage_change(self):
        """Show/hide stage-specific options."""
        stage = self.stage_var.get()

        if stage == "0":
            self.api_frame.pack(fill=tk.X, pady=5,
                                after=self.stage_var.master if hasattr(self.stage_var, 'master') else None)
        else:
            self.api_frame.pack_forget()

        if stage == "3":
            self.microdata_frame.pack(fill=tk.X, pady=5)
        else:
            self.microdata_frame.pack_forget()

    def browse_folder(self, var):
        """Browse for folder."""
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder + "/")

    def browse_path(self, var):
        """Browse for file or folder."""
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def get_selected_states(self):
        """Get list of selected states."""
        return [state for state, var in self.state_checkbuttons.items() if var.get()]

    def save_config(self):
        """Save configuration to YAML file."""
        save_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
        if not save_path:
            return

        # Build config structure
        config = {
            'generalConfig': {
                'SEED': int(self.rseed_var.get()),
                'API_KEY': self.api_var.get(),
                'YEAR': int(self.year_var.get()),
                'QTR': int(self.qtr_var.get()),
                'STATES': self.get_selected_states(),
                'SKIP_TO_MICRODATA': self.stage_var.get() == "3",
                'COMBINED_DATA': self.path_vars['combined_file'].get(),
                'NAICS6_FILE': self.path_vars['naics6_file'].get(),
            },
            'preprocessConfig': {
                'DATA_IN_FOLDER': self.data_folder_var.get(),
                'CBPDATA': self.path_vars['cbp_file'].get(),
                'IMPUTECBP': self.path_vars['impute_cbp_file'].get(),
                'QWIDATA': self.path_vars['qwi_folder'].get(),
                'QCEWDIR': self.path_vars['qcew_folder'].get(),
                'OUTPATH': self.path_vars['combined_file'].get(),
            },
            'quarterConfig': {
                'COOKS_THRESH': float(self.cooks_var.get()),
                'OUTLIER_THRESH': float(self.outlier_var.get()),
                'RSEED': int(self.rseed_var.get()),
            },
            'employmentConfig': {
                'M2EMP_NOISECOEF': float(self.m2emp_var.get()),
                'COOKS_THRESH': float(self.cooks_var.get()),
                'OUTLIER_THRESH': float(self.outlier_var.get()),
                'RSEED': int(self.rseed_var.get()),
            },
            'wageConfig': {
                'COOKS_THRESH': float(self.cooks_var.get()),
                'OUTLIER_THRESH': float(self.outlier_var.get()),
                'RSEED': int(self.rseed_var.get()),
            },
            'microdataConfig': {
                'EST_SEED': int(self.rseed_var.get()),
                'M2EMP_NOISECOEF': float(self.m2emp_var.get()),
                'GAM_SHAPE': int(self.gam_shape_var.get()),
                'GAM_SCALE': int(self.gam_scale_var.get()),
                'SUBSET_OUTPATH': self.path_vars['subset_folder'].get(),
                'OUTPATH': self.path_vars['state_folder'].get(),
            }
        }

        try:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            messagebox.showinfo("Success", f"Configuration saved to {save_path}")
            self.config_name_var.set(save_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")

    def use_defaults(self):
        """Use all default settings and run."""
        config_file = self.config_name_var.get()
        if not config_file:
            messagebox.showerror("Error", "Please specify a configuration file")
            return

        # First save the configuration
        self.save_config()

        messagebox.showinfo("Info", f"Configuration saved. Ready to process with config: {config_file}")
        # Here you would call: start(config_file) from main.py

    def customize_models(self):
        """Switch to model customization tab."""
        self.notebook.select(self.model_tab)


if __name__ == "__main__":
    root = tk.Tk()
    gui = DataConfigGUI(root)
    root.mainloop()