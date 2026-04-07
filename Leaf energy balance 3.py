import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.optimize import fsolve

def calculate_leaf_energy_balance(
        gs_top, gs_bottom, t_sky_c, t_ground_c, t_leaf_c,
        t_air_c, leaf_length_cm, wind_speed, rh_percent, p_mbar,
        g_bl_total, absorptivity_sw=0.5, emissivity=0.95
):
    SIGMA = 0.00000005673
    K_AIR = 0.0261
    A_FACTOR = 0.004

    to_k = lambda c: c + 273.15
    tk_sky, tk_ground = to_k(t_sky_c), to_k(t_ground_c)
    tk_leaf = to_k(t_leaf_c)
    tk_air = to_k(t_air_c)
    leaf_length_m = leaf_length_cm / 100

    gs_total_abs = (gs_top + gs_bottom) * absorptivity_sw

    ir_abs = emissivity * SIGMA * (tk_sky ** 4 + tk_ground ** 4)
    ir_em = -emissivity * SIGMA * (tk_leaf ** 4) * 2

    if wind_speed > 0:
        d_m = A_FACTOR * np.sqrt(abs(leaf_length_m) / wind_speed)
    else:
        d_m = 0.01

    wa = 2 * K_AIR * (tk_air - tk_leaf) / d_m

    es_mbar = lambda t_k: np.exp(52.57633 - (6790.4985 / t_k) - 5.02808 * np.log(t_k)) * 10
    
    # 1:1 Excel Parity: Sättigungsdampfdruck wird über Lufttemperatur (tk_air) berechnet
    e_sat_leaf = es_mbar(tk_air) 
    e_air = es_mbar(tk_air) * (rh_percent / 100)
    vpd_mbar = e_sat_leaf - e_air

    # 1:1 Excel Parity: Transpiration nutzt die summierte Leitfähigkeit (g_bl_total)
    tk_val = -(g_bl_total * (vpd_mbar / p_mbar)) * 44
    
    net_energy = gs_total_abs + ir_abs + ir_em + wa + tk_val

    return {
        "GS_abs": gs_total_abs,
        "IR_abs": ir_abs,
        "IR_em": ir_em,
        "IR_net": ir_abs + ir_em,
        "Convection_WA": wa,
        "Transpiration_TK": tk_val,
        "Net_Balance": net_energy
    }

class EnergyBalanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Leaf Energy Balance Simulator")

        self.input_vars = {
            "gs_top": tk.StringVar(value="1000"),
            "gs_bottom": tk.StringVar(value="200"),
            "absorptivity_sw": tk.StringVar(value="0.5"),
            "t_sky_c": tk.StringVar(value="-20"),
            "t_ground_c": tk.StringVar(value="45"),
            "t_leaf_c": tk.StringVar(value="45"),
            "t_air_c": tk.StringVar(value="38"),
            "leaf_length_cm": tk.StringVar(value="5"),
            "wind_speed": tk.StringVar(value="0.1"),
            "rh_percent": tk.StringVar(value="30"),
            "p_mbar": tk.StringVar(value="1013"),
            "g_bl_total": tk.StringVar(value="69.6") 
        }
        self.target_balance = tk.StringVar(value="0")
        self.create_widgets()

    def create_widgets(self):
        input_frame = tk.LabelFrame(self.root, text="Input variables", font=('Arial', 16, 'bold'))
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        self.input_mapping = {
            "gs_top": "Global radiation top (W/m²)",
            "gs_bottom": "Global radiation bottom (W/m²)",
            "absorptivity_sw": "Absorptivity SW (0.0 - 1.0)",
            "t_sky_c": "Sky temperature (°C)",
            "t_ground_c": "Ground temperature (°C)",
            "t_leaf_c": "Leaf temp (°C)",
            "t_air_c": "Air temperature (°C)",
            "leaf_length_cm": "Leaf length (cm)",
            "wind_speed": "Wind speed (m/s)",
            "rh_percent": "Relative humidity (%)",
            "p_mbar": "Air pressure (mbar)",
            "g_bl_total": "Stomatal conductance total (mmol/m²/s)"
        }

        row = 0
        for key, var in self.input_vars.items():
            display_name = self.input_mapping.get(key, key)
            ttk.Label(input_frame, text=display_name).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            
            if key in ["t_leaf_c", "g_bl_total"]:
                entry = tk.Entry(input_frame, textvariable=var, width=15, bg="lightgray")
            else:
                entry = ttk.Entry(input_frame, textvariable=var, width=15)
                
            entry.grid(row=row, column=1, padx=5, pady=2)
            row += 1

        ttk.Label(input_frame, text="Target Net Balance:").grid(row=row, column=0, sticky="w", padx=5, pady=10)
        ttk.Entry(input_frame, textvariable=self.target_balance, width=15).grid(row=row, column=1, padx=5, pady=10)

        calc_btn = tk.Button(input_frame, text="Calculate/Solve", font=('Arial', 16, 'bold'), bg="lightgreen", command=self.process)
        calc_btn.grid(row=row + 1, column=0, columnspan=2, pady=10)

        self.output_frame = tk.LabelFrame(self.root, text="Results", font=('Arial', 16, 'bold'))
        self.output_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")
        
        self.output_mapping = {
            "GS_abs": "Global radiation absorbed (W/m²)",
            "IR_abs": "Longwave radiation absorbed (W/m²)",
            "IR_em": "Longwave radiation emitted (W/m²)",
            "Convection_WA": "Sensible heat flux (Convection) (W/m²)",
            "Transpiration_TK": "Latent heat flux (Transpiration) (W/m²)",
            "Net_Balance": "Net Energy Balance (W/m²)"
        }
        
        self.out_labels = {}
        row_out = 0
        bold_font = ('Arial', 14, 'bold')
        
        for internal_key, display_name in self.output_mapping.items():
            ttk.Label(self.output_frame, text=display_name + ":", font=bold_font).grid(row=row_out, column=0, sticky="w", padx=5, pady=2)
            lbl = ttk.Label(self.output_frame, text="-", font=bold_font)
            lbl.grid(row=row_out, column=1, sticky="w", padx=5, pady=2)
            self.out_labels[internal_key] = lbl
            row_out += 1

        self.solver_result_lbl = ttk.Label(self.output_frame, text="", foreground="blue", font=('Arial', 10, 'bold'))
        self.solver_result_lbl.grid(row=row_out, column=0, columnspan=2, pady=10)
    
    def process(self): 
        inputs = {}
        missing_keys = []

        for key, var in self.input_vars.items():
            val = var.get().strip()
            if val == "":
                missing_keys.append(key)
                inputs[key] = None
            else:
                try:
                    inputs[key] = float(val)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid value with {key}")
                    return

        if len(missing_keys) == 0:
            self.run_direct_calculation(inputs)
        elif len(missing_keys) == 1:
            self.run_solver(inputs, missing_keys[0])
        else:
            messagebox.showerror("Error", "Leave exactly ONE field blank to solve for it.")

    def run_direct_calculation(self, inputs):
        results = calculate_leaf_energy_balance(**inputs)
        self.update_outputs(results)
        self.solver_result_lbl.config(text="")
    
    def run_solver(self, inputs, missing_key):
        try:
            target = float(self.target_balance.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid value for Target Net Balance")
            return
        
        def objective_func(x):
            test_inputs = inputs.copy()
            test_inputs[missing_key] = x[0]
            res = calculate_leaf_energy_balance(**test_inputs)
            return res["Net_Balance"] - target

        initial_guess = [20.0] if "t_" in missing_key else [100.0]
        solution, info, ier, mesg = fsolve(objective_func, initial_guess, full_output=True)

        if ier == 1:
            inputs[missing_key] = solution[0]
            final_results = calculate_leaf_energy_balance(**inputs)
            self.update_outputs(final_results)
            self.input_vars[missing_key].set(f"{solution[0]:.4f}")
            display_name = self.input_mapping.get(missing_key, missing_key)
            self.solver_result_lbl.config(text=f"Solved: {display_name} = {solution[0]:.4f}")
        else:
            messagebox.showerror("Error", f"No solution found: {mesg}")
   
    def update_outputs(self, results):
        for key, val in results.items():
            if key in self.out_labels:
                self.out_labels[key].config(text=f"{val:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyBalanceApp(root)
    root.mainloop()
