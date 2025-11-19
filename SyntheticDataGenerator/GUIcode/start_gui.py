import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import yaml
import os
import traceback
import sys

sys.path.append(os.path.abspath("./NAICS6_Pyfunctions/"))
from GeneralFunctions import *
from config_reader import check_config

#scrollable overall frame (To be used in Interactive  Model Builder)
class ScrollableFrame(ttk.Frame): #overall frame
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class DataSelectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Selection Interface")
        self.root.geometry("900x1000")

        self.selected_quarters = {}
        self.selected_states = set()
        self.quarter_checkboxes = {}
        self.state_checkboxes = {}

        self.create_widgets()

    def create_widgets(self):
        """Create all GUI widgets"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        #Prompt for what stage you are on?
        #self.create_stage_tab(notebook)

        # Tab 1: Quarter and Year Selection
        self.create_quarter_tab(notebook)

        # Tab 2: State Selection
        self.create_state_tab(notebook)

        # Tab 3: File Locations
        self.create_file_tab(notebook)

        # Bottom frame with submit button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Submit", command=self.submit).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset).pack(side=tk.RIGHT, padx=5)

    def create_stage_tab(self, notebook):
        """Create quarters and years selection tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Main: Steps")

        # Instructions
        #ttk.Label(frame,
        #          text="What stage are you on? Answer the following questions before moving on to the 'Quarter & Years' and 'States' tabs",
        #          font=("Arial", 10,"bold")).pack(pady=10)


        #ttk.Label(frame, text=f"Retrieving and combining the datasets.", font=("Arial", 10)).grid(
        #    row=0, column=0, columnspan=2, sticky="w", pady=10
        #)
        #for i, var in enumerate(["QWI","QCEW","CBP","imputed CBP employment"], start=4):
        #    include_var = tk.BooleanVar(value=False)
        #    transform_var = tk.StringVar()
        #    degree_var = tk.StringVar(value="")
        #ttk.OptionMenu(frame, self.retrieve, self.retrieve.get(), "Already Downloaded", "Need to Download").grid(row=1,
        #                                                                                                          column=1,
        #                                                                                                         sticky="w")



        #ttk.Label(frame, text="Transform response:").grid(row=1, column=0, sticky="w")
        #ttk.OptionMenu(frame, self.response_transform, self.response_transform.get(), "none", "log", "sqrt").grid(row=1,
        #                                                                                                          column=1,
        #                                                                                                          sticky="w")

        #ttk.Separator(frame, orient="horizontal").grid(row=2, column=0, columnspan=5, sticky="ew", pady=10)

        # Header
        #ttk.Label(frame, text="Variable").grid(row=3, column=0, padx=5)
        #ttk.Label(frame, text="Include").grid(row=3, column=1)
        #ttk.Label(frame, text="Transform").grid(row=3, column=2)
        #ttk.Label(frame, text="Polynomial degree").grid(row=3, column=3)

        # Variables
        for i, var in enumerate([c for c in self.df.columns if c != self.response_var], start=4):
            include_var = tk.BooleanVar(value=False)
            transform_var = tk.StringVar()
            degree_var = tk.StringVar(value="")

            # Default transformation — categorical for non-numeric
            if pd.api.types.is_numeric_dtype(self.df[var]):
                transform_var.set("none")
            else:
                transform_var.set("categorical")

            # Restore from previous session if present
            if 'variable_settings' in self.prev_state and var in self.prev_state['variable_settings']:
                prev_cfg = self.prev_state['variable_settings'][var]
                include_var.set(prev_cfg['include'])
                transform_var.set(prev_cfg['transform'])
                degree_var.set(prev_cfg['degree'])

            ttk.Label(frame, text=var).grid(row=i, column=0, sticky="w", padx=5)
            ttk.Checkbutton(frame, variable=include_var).grid(row=i, column=1, padx=5)
            transform_menu = ttk.OptionMenu(frame, transform_var, transform_var.get(), "none", "categorical", "log",
                                            "sqrt")
            transform_menu.grid(row=i, column=2, padx=5)
            poly_entry = ttk.Entry(frame, textvariable=degree_var, width=10)
            poly_entry.grid(row=i, column=3, padx=5)


    def create_quarter_tab(self, notebook):
        """Create quarters and years selection tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Quarters & Years")

        # Instructions
        ttk.Label(frame, text="Select Quarters (Q1-Q4) for Years (1998-2016)", font=("Arial", 10, "bold")).pack(pady=10)

        # Control buttons
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="Select All", command=self.select_all_quarters).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Deselect All", command=self.deselect_all_quarters).pack(side=tk.LEFT, padx=5)

        # Canvas with scrollbar for grid
        canvas_frame = ttk.Frame(frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create checkbox grid
        years = list(range(1998, 2017))
        quarters = ["Q1", "Q2", "Q3", "Q4"]

        # Header row with column select buttons
        header_frame = ttk.Frame(scrollable_frame)
        header_frame.grid(row=0, column=0, columnspan=5, sticky="ew", padx=5, pady=5)

        ttk.Label(header_frame, text="Year/Quarter", width=12).pack(side=tk.LEFT)
        for q in quarters:
            btn_frame = ttk.Frame(header_frame)
            btn_frame.pack(side=tk.LEFT, padx=2)
            ttk.Button(btn_frame, text=f"All {q}", width=8,
                       command=lambda qu=q: self.select_column_quarters(qu)).pack()

        # Data rows
        for i, year in enumerate(years, start=1):
            year_frame = ttk.Frame(scrollable_frame)
            year_frame.grid(row=i, column=0, columnspan=5, sticky="ew", padx=5, pady=2)

            # Select all for this year button
            ttk.Button(year_frame, text=f"All {year}", width=10,
                       command=lambda y=year: self.select_row_quarters(y)).pack(side=tk.LEFT, padx=2)

            for j, q in enumerate(quarters):
                var = tk.BooleanVar()
                self.selected_quarters[(year, q)] = var
                cb = ttk.Checkbutton(year_frame, variable=var, text=f"{q}")
                cb.pack(side=tk.LEFT, padx=2)
                self.quarter_checkboxes[(year, q)] = cb

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_state_tab(self, notebook):
        """Create state selection tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="States")

        ttk.Label(frame, text="Select States", font=("Arial", 10, "bold")).pack(pady=10)

        # Control buttons
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="Select All States", command=self.select_all_states).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Deselect All States", command=self.deselect_all_states).pack(side=tk.LEFT,
                                                                                                     padx=5)

        # Canvas with scrollbar
        canvas_frame = ttk.Frame(frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(canvas_frame, height=300)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # US States
        states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                  "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                  "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                  "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

        for i, state in enumerate(states):
            var = tk.BooleanVar()
            self.selected_states_vars = getattr(self, 'selected_states_vars', {})
            self.selected_states_vars[state] = var

            cb = ttk.Checkbutton(scrollable_frame, variable=var, text=state,
                                 command=lambda s=state, v=var: self.update_state_selection(s, v))
            cb.grid(row=i//10,column=i%10,)#pack(anchor=tk.W, padx=10, pady=2)
            self.state_checkboxes[state] = (cb, var)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_file_tab(self, notebook):
        """Create file location selection tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="File Locations")

        # imputedf file
        file_frame = ttk.LabelFrame(frame, text="1. ImputeDF File (CSV or TXT)", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=10)

        self.imputedf_var = tk.StringVar()
        ttk.Label(file_frame, text="File:").pack(side=tk.LEFT)
        ttk.Entry(file_frame, textvariable=self.imputedf_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_imputedf).pack(side=tk.LEFT)

        # CBP file
        cbp_frame = ttk.LabelFrame(frame, text="2. CBP File (CSV or TXT)", padding=10)
        cbp_frame.pack(fill=tk.X, padx=10, pady=10)

        self.cbp_var = tk.StringVar()
        ttk.Label(cbp_frame, text="File (optional):").pack(side=tk.LEFT)
        ttk.Entry(cbp_frame, textvariable=self.cbp_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(cbp_frame, text="Browse", command=self.browse_cbp).pack(side=tk.LEFT)

        # QWI folder
        qwi_frame = ttk.LabelFrame(frame, text="3. QWI Folder", padding=10)
        qwi_frame.pack(fill=tk.X, padx=10, pady=10)

        self.qwi_var = tk.StringVar()
        ttk.Label(qwi_frame, text="Folder (optional):").pack(side=tk.LEFT)
        ttk.Entry(qwi_frame, textvariable=self.qwi_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(qwi_frame, text="Browse", command=self.browse_qwi_folder).pack(side=tk.LEFT)

        # Census API Key
        api_frame = ttk.LabelFrame(frame, text="Census API Key", padding=10)
        api_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(api_frame, text="(Required if CBP/QWI files not provided)").pack(anchor=tk.W)
        self.api_key_var = tk.StringVar()
        ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*").pack(fill=tk.X, pady=5)

        # QCEW file
        qcew_frame = ttk.LabelFrame(frame, text="4. QCEW File (CSV or TXT)", padding=10)
        qcew_frame.pack(fill=tk.X, padx=10, pady=10)

        self.qcew_var = tk.StringVar()
        ttk.Label(qcew_frame, text="File (optional):").pack(side=tk.LEFT)
        ttk.Entry(qcew_frame, textvariable=self.qcew_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(qcew_frame, text="Browse", command=self.browse_qcew).pack(side=tk.LEFT)

    def select_all_quarters(self):
        """Select all quarters"""
        for var in self.selected_quarters.values():
            var.set(True)

    def deselect_all_quarters(self):
        """Deselect all quarters"""
        for var in self.selected_quarters.values():
            var.set(False)

    def select_row_quarters(self, year):
        """Select all quarters for a specific year"""
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if (year, q) in self.selected_quarters:
                self.selected_quarters[(year, q)].set(True)

    def select_column_quarters(self, quarter):
        """Select a specific quarter for all years"""
        for year in range(1998, 2017):
            if (year, quarter) in self.selected_quarters:
                self.selected_quarters[(year, quarter)].set(True)

    def select_all_states(self):
        """Select all states"""
        for state, (cb, var) in self.state_checkboxes.items():
            var.set(True)
            self.selected_states.add(state)

    def deselect_all_states(self):
        """Deselect all states"""
        for state, (cb, var) in self.state_checkboxes.items():
            var.set(False)
        self.selected_states.clear()

    def update_state_selection(self, state, var):
        """Update state selection set"""
        if var.get():
            self.selected_states.add(state)
        else:
            self.selected_states.discard(state)

    def browse_imputedf(self):
        """Browse for imputedf file"""
        file = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])
        if file:
            self.imputedf_var.set(file)

    def browse_cbp(self):
        """Browse for CBP file"""
        file = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])
        if file:
            self.cbp_var.set(file)

    def browse_qwi_folder(self):
        """Browse for QWI folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.qwi_var.set(folder)

    def browse_qcew(self):
        """Browse for QCEW file"""
        file = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])
        if file:
            self.qcew_var.set(file)

    def submit(self):
        """Validate and submit form"""
        # Validate quarters selected
        selected_quarters = [key for key, var in self.selected_quarters.items() if var.get()]
        if not selected_quarters:
            messagebox.showwarning("Validation", "Please select at least one quarter.")
            return

        # Validate states selected
        if not self.selected_states:
            messagebox.showwarning("Validation", "Please select at least one state.")
            return

        # Validate imputedf file
        if not self.imputedf_var.get():
            messagebox.showwarning("Validation", "Please select an imputedf file.")
            return

        # Validate API key or CBP/QWI files
        has_cbp_or_qwi = bool(self.cbp_var.get() or self.qwi_var.get())
        has_api_key = bool(self.api_key_var.get())

        if not has_cbp_or_qwi and not has_api_key:
            messagebox.showwarning("Validation", "Please provide either CBP/QWI files or a Census API key.")
            return

        # Collect data
        data = {
            "quarters": selected_quarters,
            "states": sorted(list(self.selected_states)),
            "imputedf_file": self.imputedf_var.get(),
            "cbp_file": self.cbp_var.get() or None,
            "qwi_folder": self.qwi_var.get() or None,
            "qcew_file": self.qcew_var.get() or None,
            "api_key": self.api_key_var.get() or None
        }

        messagebox.showinfo("Success",
                            f"Form submitted!\n\nQuarters: {len(selected_quarters)}\nStates: {len(self.selected_states)}\n\nCheck console for full data.")
        print("\n" + "=" * 50)
        print("SUBMITTED DATA:")
        print("=" * 50)
        print(f"Selected Quarters: {selected_quarters}")
        print(f"Selected States: {data['states']}")
        print(f"ImputeDF File: {data['imputedf_file']}")
        print(f"CBP File: {data['cbp_file']}")
        print(f"QWI Folder: {data['qwi_folder']}")
        print(f"QCEW File: {data['qcew_file']}")
        print(f"Census API Key: {'*' * len(data['api_key']) if data['api_key'] else 'Not provided'}")
        print("=" * 50 + "\n")

    def reset(self):
        """Reset all selections"""
        self.deselect_all_quarters()
        self.deselect_all_states()
        self.imputedf_var.set("")
        self.cbp_var.set("")
        self.qwi_var.set("")
        self.qcew_var.set("")
        self.api_key_var.set("")



class StartGUI:
    def __init__(self,root):
        self.root = root
        self.root.title("Synthetic Establishment-Level Data Generation")
        self.root.geometry("800x600")

        self.states=[]

        container = ScrollableFrame(self)  # scrollable y for the whole window
        container.pack(fill="both", expand=True)
        frame = container.scrollable_frame

        top = ttk.Frame(frame, padding=8)
        top.pack(side='top', fill='x')

        ttk.Label(top, text='What stage are you at?').pack(anchor='w')



class ConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Configuration Builder")
        self.root.geometry("800x600")

        self.config_data = {}
        self.entries = {}

        self.setup_ui()

    def setup_ui(self):
        # Top buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", pady=10)

        ttk.Button(button_frame, text="Load Template (config_pre2017.yaml)", command=self.load_template).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Upload Existing YAML", command=self.upload_yaml).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Validate Config", command=self.validate_config).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_yaml).pack(side="left", padx=5)

        # Scrollable content area
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def load_template(self):
        try:
            with open("config_pre2017.yaml", "r") as f:
                self.config_data = yaml.safe_load(f)
            self.build_form()
            messagebox.showinfo("Success", "Template loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load template: {e}")

    def upload_yaml(self):
        file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if not file_path:
            return
        try:
            with open(file_path, "r") as f:
                user_config = yaml.safe_load(f)

            if not self.config_data:
                self.load_template()

            # Merge user config with template
            self.merge_dicts(self.config_data, user_config)
            self.build_form()
            messagebox.showinfo("Success", "YAML loaded and merged successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YAML: {e}")

    def merge_dicts(self, base, update):
        """Recursively merges user config into template."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self.merge_dicts(base[key], value)
            else:
                base[key] = value

    def build_form(self):
        # Clear old widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        # Create input fields
        for section, values in self.config_data.items():
            ttk.Label(self.scrollable_frame, text=f"[{section}]", font=("Arial", 14, "bold")).pack(anchor="w", pady=5)
            if isinstance(values, dict):
                for key, val in values.items():
                    frame = ttk.Frame(self.scrollable_frame)
                    frame.pack(fill="x", pady=2)
                    ttk.Label(frame, text=key, width=25).pack(side="left")
                    entry = ttk.Entry(frame, width=60)
                    entry.insert(0, str(val) if val is not None else "")
                    entry.pack(side="left", padx=5)
                    self.entries[f"{section}.{key}"] = entry

    def collect_config(self):
        updated = {}
        for full_key, entry in self.entries.items():
            section, key = full_key.split(".")
            if section not in updated:
                updated[section] = {}
            val = entry.get()
            # Try to cast to int/float/None if possible
            if val.lower() == "none":
                val = None
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            updated[section][key] = val
        return updated

    def validate_config(self):
        try:
            config_dict = self.collect_config()
            temp_path = "_temp_config.yaml"
            with open(temp_path, "w") as f:
                yaml.dump(config_dict, f)
            check_config(temp_path)
            os.remove(temp_path)
            messagebox.showinfo("Validation Passed", "Configuration is valid!")
        except Exception as e:
            traceback_str = "".join(traceback.format_exception(None, e, e.__traceback__))
            messagebox.showerror("Validation Failed", traceback_str)

    def save_yaml(self):
        try:
            config_dict = self.collect_config()
            file_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
            if not file_path:
                return
            with open(file_path, "w") as f:
                yaml.dump(config_dict, f, sort_keys=False)
            messagebox.showinfo("Success", f"Configuration saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save YAML: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    #app = ConfigGUI(root)
    app = DataSelectionGUI(root)
    root.mainloop()
