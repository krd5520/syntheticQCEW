
import os
import sys
import itertools
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import threading
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
## Need seaborn and statsmodels.graphics.gofplots if plotting
matplotlib.use('TkAgg')
##test

sys.path.append(os.path.abspath("./NAICS6_Pyfunctions/"))
from GeneralFunctions import get_model, custom_predict, possible_variables

TRANSFORMS = ['none', 'log', 'sqrt', 'categorical', '2-degree polynomial','3-degree polynomial','4-degree polynomial']
YTRANSFORMS = ['none', 'log', 'sqrt']


"""
GUI for model specification. 
"""

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

#Interctive Model builder
class OLSPromptGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('OLS_prompt — interactive model builder') #title
        self.geometry('1100x700') #size

        container = ScrollableFrame(self) #scrollable y for the whole window
        container.pack(fill="both", expand=True)
        frame = container.scrollable_frame

        #initialize variables
        self.data = None
        self.response_var = None
        self.predictors = []
        self.formula_str = ''
        self.progress_fit=''
        self.progress_pred=''
        self.progress=''
        self.pred_rseed=1


        ###### UI ############
        # top frame to load CSV
        top = ttk.Frame(frame, padding=8)
        top.pack(side='top', fill='x')
        top.columnconfigure(1,weight=1)


        loadcsv_btn=ttk.Button(top, text='Load CSV', command=self.load_csv)#.pack(side='left')
        loadcsv_btn.grid(row=0,column=0,pady=5,padx=5,sticky="w")

        self.status_label=ttk.Label(top,text="Waiting for command.")
        self.status_label.grid(row=0,column=2,padx=5,pady=5,sticky="e")

        self.progress = ttk.Progressbar(top, mode="indeterminate", length=150)
        self.progress.grid(row=0,column=3,padx=5,pady=5,sticky="e")

        rseedlab=ttk.Label(top, text="Random Seed:")
        rseedlab.grid(row=1,column=0,pady=5,padx=5,sticky="w")#".pack(anchor='w')
        self.pred_rseed = tk.IntVar(value=1)
        rseedentry=ttk.Entry(top, textvariable=self.pred_rseed)
        rseedentry.grid(row=1, column=1, pady=5, padx=5, sticky="w")  # ".pack(anchor='w')

        #middle section of UI (controls is user input, output is model results)
        middle = ttk.Frame(frame)
        middle.pack(fill='both', expand=True)

        controls = ttk.Frame(middle, padding=8)
        controls.pack(side='left', fill='y')

        ttk.Label(controls, text='Potential Predictor Variables\n(always available when response is NA)').pack(anchor='w')
        self.var_listbox = tk.Listbox(controls, height=5, selectmode='extended')
        self.var_listbox.pack(fill='x')

        #self.model_file_var = tk.StringVar()
        #ttk.Label(controls, text='For manual exploration of prediction model.').pack(
        #    anchor='w')
        #ttk.Button(controls, text='Export sub-dataset as CSV', command=self.ask_save_model_file).pack(fill='x', pady=4)
        #ttk.Label(controls, textvariable=self.model_file_var, wraplength=200).pack()

        #ttk.Separator(controls).pack(fill='x', pady=6)

        ttk.Label(controls, text='Prediction model formula\n(Use step-by-step builder or formula format guide below)').pack(anchor='w')
        #ttk.Label(controls,
        #          text='  Statsmodels formula guide:\n https://www.statsmodels.org/stable/example_formulas.html)').pack(
        #    anchor='w')
        # Clickable link to statsmodels formula guide
        link_label = tk.Label(
            controls, text='https://www.statsmodels.org/stable/example_formulas.html',
            fg="blue", cursor="hand2", font=('Arial', 9, 'underline')
        )
        link_label.pack(anchor="w", padx=10)
        link_label.bind(
            "<Button-1>",
            lambda e: webbrowser.open_new("https://www.statsmodels.org/stable/example_formulas.html")
        )

        self.direct_formula_text = tk.Text(controls, height=3, width=40)
        self.direct_formula_text.pack()
        ttk.Button(controls, text='Use direct model formula', command=self.use_direct_formula).pack(fill='x', pady=4)
        # Button to open step-by-step popout
        ttk.Button(controls, text="Open step-by-step model formula builder", command=self.open_step_builder).pack(
            anchor='w', pady=(6, 0))

        ttk.Separator(controls).pack(fill='x', pady=6)

        ttk.Label(controls, text='Diagnostics / thresholds').pack(anchor='w')
        self.cooks_var = tk.DoubleVar(value=1.0)
        ttk.Entry(controls, textvariable=self.cooks_var).pack(fill='x')
        ttk.Label(controls, text="Cook's Distance Threshold").pack(anchor='w')
        self.outlier_var = tk.DoubleVar(value=3.0)
        ttk.Entry(controls, textvariable=self.outlier_var).pack(fill='x')
        ttk.Label(controls, text='Studentized Residual Threshold').pack(anchor='w')

        self.diagnostic_folder = tk.StringVar()
        ttk.Button(controls, text='Choose diagnostic plot save location (optional)', command=self.choose_diag_folder).pack(fill='x', pady=4)
        ttk.Label(controls, textvariable=self.diagnostic_folder, wraplength=200).pack()

        self.fit_btn = ttk.Button(controls, text='Fit model now', command=self.processing_fitting, state='disabled')
        self.fit_btn.pack(fill='x', pady=8)


        outputs = ttk.Frame(middle, padding=8)
        outputs.pack(side='left', fill='both', expand=True)

        ttk.Label(outputs, text='Constructed formula:').pack(anchor='w')
        self.formula_label = ttk.Label(outputs, text='', wraplength=700, background='white')
        self.formula_label.pack(fill='x', pady=4)

        ttk.Label(outputs, text='OLS Summary').pack(anchor='w')
        self.summary_text = tk.Text(outputs, height=12)
        self.summary_text.pack(fill='both', expand=False)

        #self.diagnostic_button = ttk.Button(outputs, text="Show Diagnostic Plots", command=self.show_diagnostic_plots, state='diabled')
        #self.diagnostic_button.pack(pady=20)


        self.plot_frame = ttk.Frame(outputs)
        self.plot_frame.pack(fill='both', expand=True)

        self.proceed_btn = ttk.Button(outputs, text='Proceed to custom predictions', command=self.processing_prediction, state='disabled')
        self.proceed_btn.pack(fill='x', pady=6)


        self.pred_config = {}
        self.interactions = []
        self.fitted_model = None
        self.prev_step_state = {}  # store previous step builder state

    def update_fit_button_state(self):
        if self.formula_str:
            self.fit_btn.config(state='normal')
        else:
            self.fit_btn.config(state='disabled')

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[('CSV files','*.csv'), ('All files','*.*')])
        if not path:
            return
        df = pd.read_csv(path,index_col=0,low_memory=False)
        cols = list(df.columns)
        resp = simpledialog.askstring('Response variable', f'Columns found:\n{cols}\n\nEnter the response variable:')
        if resp not in cols:
            messagebox.showerror('Error', f'{resp} not in columns')
            return
        possible_vars=possible_variables(df,resp)
        possible_vars.append(resp)
        self.data = df[possible_vars]
        self.response_var = resp
        others = [c for c in possible_vars if c != resp]
        self.var_listbox.delete(0, 'end')
        for c in others:
            self.var_listbox.insert('end', c)

    def ask_save_model_file(self):
        if self.data is None or self.response_var is None:
            messagebox.showwarning('No data', 'Load a dataset first')
            return
        subset = self.data[self.data[self.response_var].notna()].copy()
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path:
            return
        subset.to_csv(path, index=False)
        self.model_file_var.set(f'Saved MODEL_FILE: {path} (subset without NA {self.response_var})')

    def check_transform(self,varstr):
        save_varstr=varstr
        varstr = varstr.replace("np.sqrt(", "").replace("np.log(", "").replace("C(","").replace(")", "").strip()
        if varstr in self.data.columns:
            return save_varstr
        else:
            if "poly" in varstr:
                internalvar=varstr.replace("poly","").replace("(","")
                internalvar_split=internalvar.split(",")
                internalvar=internalvar_split[0].strip()
                if internalvar in self.data.columns:
                    return save_varstr
                else:
                    print("IDK what happened 1")
                    return save_varstr
            elif ':' in varstr:
                splitvars=varstr.split(":",1)
                var1=splitvars[0].replace("np.sqrt(","").replace("np.log(","").replace("C(","").replace(")","").strip()
                var2=splitvars[1].replace("np.sqrt(","").replace("np.log(","").replace("C(","").replace(")","").strip()
                if var1 in self.data.columns and var2 in self.data.columns:
                    return save_varstr
                else:
                    print("IDK what happened 2")
                    return save_varstr
            elif '*' in varstr:
                splitvars=varstr.split("*",1)
                var1=splitvars[0].replace("np.sqrt(","").replace("np.log(","").replace("C(","").replace(")","").strip()
                var2=splitvars[1].replace("np.sqrt(","").replace("np.log(","").replace("C(","").replace(")","").strip()
                if var1 in self.data.columns and var2 in self.data.columns:
                    return save_varstr
                else:
                    print("IDK what happened 3")
                    return save_varstr
            elif "[" in varstr:
                internalvar=varstr.split("[",1)[0]
                if internalvar in self.data.columns:
                    return save_varstr
                else:
                    print("IDK what happened 4")
                    return save_varstr
            else:
                print(varstr)
                print("IDK what happened 5")
                return save_varstr


    def use_direct_formula(self):
        text = self.direct_formula_text.get('1.0','end').strip()
        if not text:
            messagebox.showwarning('Empty', 'Enter a formula')
            return
        init_vars=re.split(r'[~,+]+',text)
        pred_vars=[x.strip() for x in init_vars[1:]]
        if self.response_var!=init_vars[0].strip():
            yvar=self.check_transform(init_vars[0].strip())

        pred_vars_str=""
        for elem in pred_vars:
            pred_vars_str=pred_vars_str+self.check_transform(elem)
        self.formula_str = text
        self.formula_label.config(text=self.formula_str)
        self.update_fit_button_state()

    def open_step_builder(self):
        """Open the interactive variable selection dialog."""
        if self.data is None or self.response_var is None:
            messagebox.showwarning('No data', 'Load data first')
            return

        dialog = VariableSelectionDialog(self, self.data, self.response_var, self.prev_step_state)
        self.wait_window(dialog)

        if dialog.formula_str:
            # Save user selections for next time
            self.formula_str = dialog.formula_str
            self.prev_step_state = dialog.save_state()

            # Update the main GUI textboxes and labels
            self.formula_label.config(text=self.formula_str)
            self.direct_formula_text.delete('1.0', 'end')
            self.direct_formula_text.insert('1.0', self.formula_str)
            self.update_fit_button_state()



    def build_formula_from_ui(self, formula, state):
        self.formula_str = formula
        self.direct_formula_text.delete('1.0','end')
        self.direct_formula_text.insert('1.0', formula)
        self.formula_label.config(text=self.formula_str)
        self.update_fit_button_state()
        self.prev_step_state = state

    def _term_for_var(self, var, conf):
        if conf.get('transform') == 'categorical': #conf.get('categorical') or
            #ref = conf.get('reference')
            #if ref:
            #    return f'C({var}, Treatment(reference="{ref}"))'
            return f'C({var})'
        t = conf.get('transform','none')
        #poly = conf.get('poly',0)
        base = var
        if t == 'log':
            base = f'np.log({base})'
        elif t == 'sqrt':
            base = f'np.sqrt({base})'
        elif t=='2-degree polynomial':#poly and poly > 1:
            base=f'poly({base},degree=2)'
        elif t=='3-degree polynomial':#poly and poly > 1:
            base=f'poly({base},degree=3)'
        elif t=='4-degree polynomial':
            base = f'poly({base},degree=4)'
        elif t=='none':
            base=f'{base}'
        return base

    def choose_diag_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.diagnostic_folder.set(d)

    def fit_model(self):
        if not self.formula_str:
            messagebox.showwarning('No formula', 'Build or enter a formula first')
            return
        if self.data is None:
            messagebox.showwarning('No data', 'Load data first')
            return
            # call get_model
        cooks = float(self.cooks_var.get())
        outthr = float(self.outlier_var.get())
        diag = None
        if self.diagnostic_folder.get():
            # create a temporary filename
            diag = os.path.join(self.diagnostic_folder.get(), 'diagnostic_plots.png')

        try:
            model, outliertext, summaryobj, fittedvals, residvals = get_model(self.data, self.formula_str, float(self.cooks_var.get()),
                                           float(self.outlier_var.get()), self.diagnostic_folder.get() or None, True,True)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Model error', f'Error fitting model: {e}'))
            return

            # Store model and output text safely
        self.fitted_model = model
        summary_info = (model, outliertext, summaryobj, fittedvals, residvals)

        self.after(0, lambda: self.update_after_fit(summary_info))


    def _show_diagnostic_plots(self, summary_info):
        model, outliertext, summaryobj, fitted, resid=summary_info
        # create figure similar to save_diagnostic_plots
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.3))
        fig.tight_layout()
        try:
            from statsmodels.graphics.gofplots import qqplot
            qqplot(resid, line='s', ax=axes[0],markersize=1)
            axes[0].set_title('Residual Q-Q Plot')
            import seaborn as sns
            sns.residplot(x=fitted, y=resid, lowess=True, line_kws={'color': 'red', 'lw': 1},
                          scatter_kws={'alpha': 0.4, 's': 1}, ax=axes[1])
            axes[1].axhline(0, color='grey', linewidth=1)
            axes[1].set_xlabel('Fitted')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title('Fitted vs. Residuals')
            for i in [0,1]:
                axes[i].tick_params(axis="both",which="major",labelsize=8)
                axes[i].tick_params(axis="both",which="minor",labelsize=6)
        except Exception as e:
            axes[0].text(0.5, 0.5, 'Could not create plots: ' + str(e))
        for child in self.plot_frame.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)


    def processing_prediction(self):
        if not self.fitted_model:
            return
        #self.pred_rseed = simpledialog.askinteger('RSEED', 'Enter RSEED (optional)')
        self.progress.start(10)
        self.status_label.config(text="Predicting missing "+self.response_var+" values from model...")
        threading.Thread(target=self.predict_response).start()

    def processing_fitting(self):
        self.progress.start(10)
        self.status_label.config(text="Fitting model...")
        threading.Thread(target=self.fit_model).start()

    def update_after_fit(self, summary_info):
        """Runs in main thread to update GUI after fitting."""
        model, outliertext, summaryobj, fittedvals, residvals = summary_info

        # Update summary text
        try:
            s = summaryobj
            self.summary_text.delete('1.0', 'end')
            self.summary_text.insert('1.0', outliertext[0] + "\n")
            self.summary_text.insert('end', outliertext[1] + "\n")
            self.summary_text.insert('end', outliertext[2] + "\n")
            self.summary_text.insert('end', s)
            #s = model.summary2()
            #s_df = s.tables[1]
            #self.summary_text.insert('end', s_df.to_string())
        except Exception:
            self.summary_text.insert('1.0', 'Could not render model summary.')
        # Show diagnostic plots (this must run on main thread)
        self._show_diagnostic_plots(summary_info)
        #except Exception:
        #    self.summary_text.insert('end', 'Could not render model diagnostic plots.')

        # Enable proceed button and stop spinner
        self.proceed_btn.config(state='normal')
        self.done_processing_fit()

    def done_processing_fit(self):
        self.progress.stop()
        self.status_label.config(text="Done fitting model. Diagnostic plots will appear soon.")

    def done_processing_pred(self):
        self.progress.stop()
        self.status_label.config(text="Done predicting.")

    def predict_response(self):
        pred, se = custom_predict(self.data, self.fitted_model, self.pred_rseed.get())
        print(len(pred))
        print(len(se))
        print(len(self.data))
        outdf = self.data.copy()
        outdf['prediction'] = pred
        outdf['se_fit'] = se
        PreviewWindow(self, outdf)
        self.after(0, self.done_processing_pred)

    def proceed_prompt(self):
        if not self.fitted_model:
            return
        self.pred_rseed = simpledialog.askinteger('RSEED', 'Enter RSEED (optional)')



# =============================================================
# Step-by-step Variable Selection Dialog
# =============================================================
class VariableSelectionDialog(tk.Toplevel):
    """Popup dialog to configure variables, transformations, and interactions."""

    def __init__(self, parent, df, response_var, prev_state):
        super().__init__(parent)
        self.parent = parent
        self.df = df
        self.response_var = response_var
        self.prev_state = prev_state or {}
        self.formula_str = None

        self.title("Step-by-step Model Builder")
        self.geometry("850x700")

        # State holders
        self.variable_settings = {}
        self.selected_interactions = self.prev_state.get('interactions', [])
        self.response_transform = tk.StringVar(value=self.prev_state.get('response_transform', 'none'))

        # Scrollable container
        container = ttk.Frame(self)
        canvas = tk.Canvas(container)
        scrollbar_y = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set)
        container.pack(fill="both", expand=True)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_y.pack(side="right", fill="y")

        self.scrollable_frame = scrollable_frame
        self._build_interface()

    def _build_interface(self):
        """Builds and optionally restores all variable widgets."""
        frame = self.scrollable_frame


        ttk.Label(frame, text=f"Response variable: {self.response_var}", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=10
        )

        ttk.Label(frame, text="Transform response:").grid(row=1, column=0, sticky="w")
        ttk.OptionMenu(frame, self.response_transform, self.response_transform.get(), "none", "log", "sqrt").grid(row=1, column=1, sticky="w")

        ttk.Separator(frame, orient="horizontal").grid(row=2, column=0, columnspan=5, sticky="ew", pady=10)

        # Header
        ttk.Label(frame, text="Variable").grid(row=3, column=0, padx=5)
        ttk.Label(frame, text="Include").grid(row=3, column=1)
        ttk.Label(frame, text="Transform").grid(row=3, column=2)
        ttk.Label(frame, text="Polynomial degree").grid(row=3, column=3)

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
            transform_menu = ttk.OptionMenu(frame, transform_var, transform_var.get(), "none", "categorical", "log", "sqrt")
            transform_menu.grid(row=i, column=2, padx=5)
            poly_entry = ttk.Entry(frame, textvariable=degree_var, width=10)
            poly_entry.grid(row=i, column=3, padx=5)

            # Disable degree if categorical
            def update_poly_state(var_ref=transform_var, entry_ref=poly_entry, degree_ref=degree_var):
                if var_ref.get() == "categorical":
                    entry_ref.config(state="disabled")
                    degree_ref.set("")
                else:
                    entry_ref.config(state="normal")

            transform_var.trace_add("write", lambda *_, tv=transform_var, pe=poly_entry, pd=degree_var: update_poly_state(tv, pe, pd))
            update_poly_state(transform_var, poly_entry, degree_var)

            self.variable_settings[var] = {"include": include_var, "transform": transform_var, "degree": degree_var}

        ttk.Button(frame, text="Edit interaction terms", command=self._open_interactions_window).grid(
            row=len(self.df.columns) + 5, column=0, columnspan=3, pady=10
        )
        ttk.Button(frame, text="Build formula", command=self._build_formula).grid(
            row=len(self.df.columns) + 6, column=0, columnspan=3, pady=10
        )

    def _open_interactions_window(self):
        """Open dialog for selecting two-way interaction terms."""
        selected_vars = [v for v, vals in self.variable_settings.items() if vals["include"].get()]

        if not selected_vars:
            messagebox.showwarning("No variables", "Select variables before adding interactions.")
            return

        inter_popup = tk.Toplevel(self)

        inter_popup.title("Add Interaction Terms")
        sizestrx=str(int(min(len(selected_vars)*75+150,1800)))
        sizestry=str(int(min(len(selected_vars)*50+150,900)))

        inter_popup.geometry(sizestrx+"x"+sizestry)

        ttk.Label(inter_popup, text="Select interaction terms:", font=("Arial", 10, "bold")).pack(pady=10)
        inter_frame = ttk.Frame(inter_popup)
        inter_frame.pack(fill="both", expand=True)

        interaction_vars = []
        row_counter=0
        col_counter=0
        for i in range(len(selected_vars)):
            for j in range(i + 1, len(selected_vars)):
                pair = f"{selected_vars[i]}:{selected_vars[j]}"
                var_check = tk.BooleanVar(value=pair in self.selected_interactions)
                checkbut=ttk.Checkbutton(inter_frame, text=pair, variable=var_check)
                checkbut.grid(row=row_counter,column=col_counter,sticky='w',padx=5,pady=5)#    .pack(anchor="w")
                row_counter=row_counter+1
                if row_counter>10:
                    row_counter=0
                    col_counter=col_counter+1
                interaction_vars.append((pair, var_check))

        def save_interactions():
            self.selected_interactions = [p for p, var in interaction_vars if var.get()]
            inter_popup.destroy()

        ttk.Button(inter_popup, text="Save Interactions", command=save_interactions).pack(pady=10)

    def _build_formula(self):
        """Construct final formula string in R-style, add transformed columns to data."""
        resp = self.response_var
        rtrans = self.response_transform.get()

        if rtrans == "log":
            lhs = f"np.log({resp})"
        elif rtrans == "sqrt":
            lhs = f"np.sqrt({resp})"
        else:
            lhs = resp

        rhs_terms = []
        transformed_vars = {}  # maps original -> new transformed name

        for var, vals in self.variable_settings.items():
            if vals["include"].get():
                trans = vals["transform"].get()
                deg = vals["degree"].get()

                transformed_name = var  # by default, no change

                # --- Handle transformations and create new columns if needed ---
                #if trans == "log":
                #    new_name = f"log_{var}"
                #    if new_name not in self.df.columns:
                #        self.df[new_name] = np.log(self.df[var])
                #   transformed_name = new_name
                #elif trans == "sqrt":
                #    new_name = f"sqrt_{var}"
                #    if new_name not in self.df.columns:
                #        self.df[new_name] = np.sqrt(self.df[var])
                #    transformed_name = new_name

                transformed_vars[var] = transformed_name

                # --- Build RHS term ---
                if trans == "categorical":
                    rhs_terms.append(f"C({var})")
                elif deg.isdigit() and int(deg) > 1:
                    rhs_terms.append(f"poly({transformed_name},{deg})")
                else:
                    rhs_terms.append(transformed_name)

        # --- Handle interactions using transformed variable names ---
        for inter in self.selected_interactions:
            a, b = inter.split(':')
            a_name = transformed_vars.get(a, a)
            b_name = transformed_vars.get(b, b)
            rhs_terms.append(f"{a_name}:{b_name}")

        # --- Final formula assembly ---
        formula = f"{lhs} ~ {' + '.join(rhs_terms)}" if rhs_terms else f"{lhs} ~ 1"

        self.formula_str = formula
        self.destroy()

    def save_state(self):
        """Save the dialog's current state for reuse."""
        return {
            "response_transform": self.response_transform.get(),
            "variable_settings": {
                var: {
                    "include": vals["include"].get(),
                    "transform": vals["transform"].get(),
                    "degree": vals["degree"].get(),
                }
                for var, vals in self.variable_settings.items()
            },
            "interactions": self.selected_interactions,
        }



class PredictorConfigDialog(tk.Toplevel):
    def __init__(self, parent, varname, df):
        super().__init__(parent)
        self.result = None
        self.title(f'Configure predictor: {varname}')
        ttk.Label(self, text=f'Variable: {varname}').pack(pady=4)

        # Always allow treating as categorical
        ttk.Label(self, text='Choose transformation type').pack()
        self.trans = tk.StringVar(value='none')
        ttk.Combobox(self, values=TRANSFORMS, textvariable=self.trans, state='readonly').pack()
        ttk.Button(self, text='OK', command=self.on_ok).pack(pady=4)

        self.ref_var = tk.StringVar()
        #self.poly = tk.IntVar(value=0)

        def show_options(*args):
            #if self.trans.get() == 'categorical':
            #    for w in self.winfo_children():
            #        if isinstance(w, ttk.Label) or isinstance(w, ttk.Combobox):
            #            w.destroy()
            #    #ttk.Label(self, text=f'Select {varname}').pack()
            #    #levels = sorted(df[varname].dropna().unique().tolist())
            #    #ttk.Label(self, text='Select reference level').pack()
            #    #combo = ttk.Combobox(self, values=levels, textvariable=self.ref_var, state='readonly')
            #    #combo.pack()
            #    ttk.Button(self, text='OK', command=self.on_ok_cat).pack(pady=4)
            #else:
            x=1
            if x==1:
                for w in self.winfo_children():
                    if isinstance(w, ttk.Label) or isinstance(w, ttk.Combobox) or isinstance(w, ttk.Spinbox):
                        w.destroy()
                ttk.Label(self, text='Transform').pack()
                trans_box = ttk.Combobox(self, values=TRANSFORMS, textvariable=self.trans, state='readonly')
                #trans_box.pack()
                #ttk.Label(self, text='Polynomial degree (0/1/2/3)').pack()
                #spin = ttk.Spinbox(self, from_=0, to=3, textvariable=self.poly)
                #spin.pack()
                ttk.Button(self, text='OK', command=self.on_ok).pack(pady=4)
        #if self.trans.get() in ['none','sqrt','log']:
        #    self.trans.trace_add('write', show_options)

    #def on_ok_cat(self):
    #    ref = self.ref_var.get()
    #    self.result = {'transform': 'categorical', 'reference': ref, 'categorical': False}
    #    self.destroy()

    def on_ok(self):
        self.result = {'transform': self.trans.get(),
                       #'poly': int(self.poly.get()),
                       'categorical': False}
        self.destroy()

class PreviewWindow(tk.Toplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.title('Predictions preview')
        text = tk.Text(self, height=20)
        text.pack(fill='both', expand=True)
        text.insert('1.0', df.head(20).to_string())


if __name__ == '__main__':
    app = OLSPromptGUI()
    app.mainloop()
    # --- Example usage ---
    #    data = pd.DataFrame({
    #        "y": [10, 12, 15],
    #        "x1": [1.0, 2.0, 3.0],
    #        "x2": [5.0, 6.0, 8.0],
    #        "x3": ["A", "B", "C"]
    #    })
    #
    #    root = tk.Tk()
    #    root.title("Main App Window")
    #    root.geometry("400x200")

