"""Tkinter desktop application for the Python MCR-ALS toolbox."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from dataclasses import fields
from pathlib import Path
from functools import partial
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import numpy as np
from scipy.io import savemat  # type: ignore[import-untyped]

from .gui_state import (
    GUIState,
    KineticModelInput,
    build_options,
    create_initial_estimate,
    format_matrix,
    kinetic_model_from_input,
    load_matrix,
    state_from_dict,
    state_to_dict,
)
from .solver import MCRALSResult, mcr_als

MATRIX_FILETYPES = [
    ("Numeric matrices", "*.csv *.tsv *.txt *.dat *.npy *.npz *.mat *.xlsx"),
    ("CSV", "*.csv"),
    ("NumPy", "*.npy *.npz"),
    ("MATLAB", "*.mat"),
    ("Excel", "*.xlsx"),
    ("All files", "*.*"),
]


class KineticModelDialog:
    """Modal editor for one kinetic hard model."""

    _FIELDS = (
        ("name", "Model name", 1),
        ("reaction_orders", "Reaction orders (reactions x species)", 3),
        ("stoichiometry", "Stoichiometry (species x reactions)", 3),
        ("initial_rate_constants", "Initial rate constants", 2),
        ("initial_concentrations", "Initial concentrations", 3),
        ("time", "Time vector(s)", 3),
        (
            "component_mapping",
            "ALS component -> species (0=ignore, species are 1-based)",
            2,
        ),
        ("colored_mask", "Colored species mask (0/1)", 2),
        ("experiment_mask", "Experiment mask (0/1, blank=all)", 2),
    )

    def __init__(
        self,
        parent: tk.Misc,
        model: KineticModelInput,
        on_save: Callable[[KineticModelInput], None],
    ) -> None:
        self.on_save = on_save
        self.window = tk.Toplevel(parent)
        self.window.title("Kinetic model")
        self.window.geometry("780x760")
        self.window.minsize(620, 560)
        self.window.transient(parent.winfo_toplevel())
        self.window.grab_set()

        container = ttk.Frame(self.window)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas, padding=12)
        body.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas_window = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(canvas_window, width=event.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(
            body,
            text=(
                "Enter matrices directly (spaces/commas, semicolons/newlines) or "
                "browse to a file. Use file.mat::variable or file.npz::key to "
                "select a named array."
            ),
            wraplength=700,
            justify="left",
        ).pack(fill="x", pady=(0, 10))

        self.texts: dict[str, tk.Text] = {}
        for key, label, height in self._FIELDS:
            group = ttk.LabelFrame(body, text=label, padding=(8, 5))
            group.pack(fill="x", pady=4)
            text = tk.Text(group, height=height, wrap="none", undo=True)
            text.insert("1.0", str(getattr(model, key)))
            text.pack(side="left", fill="both", expand=True)
            if key != "name":
                ttk.Button(
                    group,
                    text="Browse...",
                    command=partial(self._browse, key),
                ).pack(side="right", padx=(6, 0), anchor="n")
            self.texts[key] = text

        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(12, 4))
        ttk.Button(buttons, text="Cancel", command=self.window.destroy).pack(
            side="right"
        )
        ttk.Button(buttons, text="Save model", command=self._save).pack(
            side="right", padx=8
        )
        self.texts["name"].focus_set()

    def _browse(self, field: str) -> None:
        path = filedialog.askopenfilename(
            parent=self.window, filetypes=MATRIX_FILETYPES
        )
        if path:
            widget = self.texts[field]
            widget.delete("1.0", "end")
            widget.insert("1.0", path)

    def _save(self) -> None:
        values = {
            key: self.texts[key].get("1.0", "end").strip()
            for key, _label, _height in self._FIELDS
        }
        candidate = KineticModelInput(**values)
        try:
            kinetic_model_from_input(candidate)
        except Exception as exc:
            messagebox.showerror("Invalid kinetic model", str(exc), parent=self.window)
            return
        self.on_save(candidate)
        self.window.destroy()


class MCRALSGui:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MCR-ALS Toolbox for Python")
        self.root.geometry("1240x860")
        self.root.minsize(1000, 700)
        self._defaults = GUIState()
        self.vars: dict[str, tk.Variable] = {}
        for item in fields(GUIState):
            if item.name == "kinetic_models":
                continue
            value = getattr(self._defaults, item.name)
            variable: tk.Variable
            if isinstance(value, bool):
                variable = tk.BooleanVar(root, value=value)
            else:
                variable = tk.StringVar(root, value=str(value))
            self.vars[item.name] = variable
        self.kinetic_models: list[KineticModelInput] = []
        self.result: MCRALSResult | None = None
        self.result_data: np.ndarray | None = None
        self.result_initial: np.ndarray | None = None
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._running = False

        self._build_menu()
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.data_tab = ttk.Frame(self.notebook, padding=12)
        self.basic_tab = ttk.Frame(self.notebook, padding=12)
        self.advanced_tab = ttk.Frame(self.notebook, padding=12)
        self.results_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.data_tab, text="Data & experiments")
        self.notebook.add(self.basic_tab, text="Basic constraints")
        self.notebook.add(self.advanced_tab, text="Advanced models")
        self.notebook.add(self.results_tab, text="Run & results")
        self._build_data_tab()
        self._build_basic_tab()
        self._build_advanced_tab()
        self._build_results_tab()
        self.root.after(100, self._poll_events)

    def _var(self, name: str) -> tk.Variable:
        return self.vars[name]

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(
            label="Open data matrix...", command=lambda: self._browse_var("data_path")
        )
        file_menu.add_command(
            label="Open initial estimate...",
            command=lambda: self._browse_var("initial_path"),
        )
        file_menu.add_separator()
        file_menu.add_command(label="Load configuration...", command=self._load_config)
        file_menu.add_command(label="Save configuration...", command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Export NPZ...", command=self._export_npz)
        file_menu.add_command(label="Export MATLAB MAT...", command=self._export_mat)
        file_menu.add_command(label="Export CSV folder...", command=self._export_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menu.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About", command=self._about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu)

    def _source_row(
        self,
        parent: tk.Misc,
        row: int,
        label: str,
        variable_name: str,
        *,
        width: int = 70,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self._var(variable_name), width=width).grid(
            row=row, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Button(
            parent,
            text="Browse...",
            command=lambda: self._browse_var(variable_name),
        ).grid(row=row, column=2, pady=4)
        parent.columnconfigure(1, weight=1)

    def _browse_var(self, variable_name: str) -> None:
        path = filedialog.askopenfilename(parent=self.root, filetypes=MATRIX_FILETYPES)
        if path:
            self._var(variable_name).set(path)

    def _build_data_tab(self) -> None:
        files = ttk.LabelFrame(self.data_tab, text="Matrices", padding=10)
        files.pack(fill="x")
        self._source_row(files, 0, "Data matrix", "data_path")
        ttk.Label(files, text="Initial estimate method").grid(
            row=1, column=0, sticky="w", pady=4
        )
        ttk.Combobox(
            files,
            textvariable=self._var("initial_method"),
            values=("file", "simplisma", "efa"),
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky="w", padx=6)
        self._source_row(files, 2, "Initial estimate file", "initial_path")
        estimate_controls = ttk.Frame(files)
        estimate_controls.grid(row=3, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(estimate_controls, text="Components").pack(side="left")
        ttk.Entry(
            estimate_controls, textvariable=self._var("components"), width=7
        ).pack(side="left", padx=(5, 18))
        ttk.Label(estimate_controls, text="SIMPLISMA noise (%)").pack(side="left")
        ttk.Entry(
            estimate_controls,
            textvariable=self._var("pure_noise_percent"),
            width=8,
        ).pack(side="left", padx=5)

        partitions = ttk.LabelFrame(
            self.data_tab, text="Multi-experiment partitions", padding=10
        )
        partitions.pack(fill="x", pady=10)
        ttk.Label(
            partitions,
            text=(
                "Leave lengths blank for one block. Presence is one 0/1 row per "
                "row block and one column per component."
            ),
            wraplength=1050,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        for row, (label, name) in enumerate(
            (
                ("Row-block lengths", "row_lengths"),
                ("Column-block lengths", "column_lengths"),
                ("Component presence", "presence"),
            ),
            start=1,
        ):
            ttk.Label(partitions, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(partitions, textvariable=self._var(name)).grid(
                row=row, column=1, sticky="ew", padx=6, pady=3
            )
        partitions.columnconfigure(1, weight=1)

        weighted = ttk.LabelFrame(
            self.data_tab, text="Weighted MLPCA preprocessing", padding=10
        )
        weighted.pack(fill="x")
        ttk.Checkbutton(
            weighted, text="Enable weighting", variable=self._var("weighted_enabled")
        ).grid(row=0, column=0, sticky="w")
        self._source_row(
            weighted, 1, "Standard-deviation matrix", "standard_deviations_path"
        )
        settings = ttk.Frame(weighted)
        settings.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(settings, text="Convergence limit").pack(side="left")
        ttk.Entry(
            settings,
            textvariable=self._var("weighted_convergence_limit"),
            width=13,
        ).pack(side="left", padx=(5, 18))
        ttk.Label(settings, text="Maximum iterations").pack(side="left")
        ttk.Entry(
            settings,
            textvariable=self._var("weighted_max_iterations"),
            width=12,
        ).pack(side="left", padx=5)

        actions = ttk.Frame(self.data_tab)
        actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="Preview inputs", command=self._preview_inputs).pack(
            side="left"
        )
        ttk.Button(
            actions,
            text="Save generated estimate...",
            command=self._save_generated_estimate,
        ).pack(side="left", padx=8)
        self.preview_label = ttk.Label(actions, text="No inputs loaded")
        self.preview_label.pack(side="left", padx=12)

    def _build_basic_tab(self) -> None:
        iteration = ttk.LabelFrame(
            self.basic_tab, text="Iteration and normalization", padding=10
        )
        iteration.pack(fill="x")
        for label, name, width in (
            ("Maximum iterations", "max_iterations", 8),
            ("Sigma tolerance (%)", "tolerance", 10),
            ("Divergence limit", "divergence_limit", 8),
        ):
            ttk.Label(iteration, text=label).pack(side="left", padx=(0, 5))
            ttk.Entry(iteration, textvariable=self._var(name), width=width).pack(
                side="left", padx=(0, 18)
            )
        ttk.Label(iteration, text="Spectral normalization").pack(side="left")
        ttk.Combobox(
            iteration,
            textvariable=self._var("normalization"),
            values=("none", "maximum", "euclidean", "sum"),
            state="readonly",
            width=12,
        ).pack(side="left", padx=5)

        modes = ttk.Frame(self.basic_tab)
        modes.pack(fill="both", expand=True, pady=10)
        modes.columnconfigure((0, 1), weight=1, uniform="modes")
        concentration = ttk.LabelFrame(modes, text="Concentration mode", padding=10)
        spectra = ttk.LabelFrame(modes, text="Spectral mode", padding=10)
        concentration.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        spectra.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self._build_mode_constraints(concentration, "c")
        self._build_mode_constraints(spectra, "s")

        closure = ttk.LabelFrame(self.basic_tab, text="Closure", padding=10)
        closure.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(
            closure, text="Enable closure", variable=self._var("closure_enabled")
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(closure, text="Mode").grid(row=0, column=1, sticky="e")
        ttk.Combobox(
            closure,
            textvariable=self._var("closure_mode"),
            values=("concentration", "spectra"),
            state="readonly",
            width=14,
        ).grid(row=0, column=2, padx=5)
        self._closure_condition_row(closure, 1, second=False)
        ttk.Checkbutton(
            closure,
            text="Enable second disjoint group",
            variable=self._var("closure_second_enabled"),
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._closure_condition_row(closure, 3, second=True)

    def _build_mode_constraints(self, frame: ttk.LabelFrame, mode: str) -> None:
        prefix = "concentration" if mode == "c" else "spectral"
        ttk.Checkbutton(
            frame,
            text="Nonnegativity",
            variable=self._var(f"nonnegative_{mode}_enabled"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=self._var(f"nonnegative_{mode}_algorithm"),
            values=("truncate", "nnls", "fnnls"),
            state="readonly",
            width=11,
        ).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(frame, text="Component mask").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self._var(f"nonnegative_{mode}_mask")).grid(
            row=1, column=1, sticky="ew", padx=5, pady=4
        )

        ttk.Separator(frame).grid(row=2, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Checkbutton(
            frame,
            text="Unimodality",
            variable=self._var(f"unimodality_{mode}_enabled"),
        ).grid(row=3, column=0, sticky="w")
        controls = ttk.Frame(frame)
        controls.grid(row=3, column=1, sticky="w")
        ttk.Label(controls, text="Tol.").pack(side="left")
        ttk.Entry(
            controls,
            textvariable=self._var(f"unimodality_{mode}_tolerance"),
            width=7,
        ).pack(side="left", padx=3)
        ttk.Label(controls, text="Mode").pack(side="left", padx=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self._var(f"unimodality_{mode}_mode"),
            values=("0", "1", "2"),
            state="readonly",
            width=4,
        ).pack(side="left", padx=3)
        ttk.Label(frame, text="Component mask").grid(row=4, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self._var(f"unimodality_{mode}_mask")).grid(
            row=4, column=1, sticky="ew", padx=5, pady=4
        )

        ttk.Separator(frame).grid(row=5, column=0, columnspan=3, sticky="ew", pady=8)
        self._source_row(
            frame,
            6,
            "Equality/bound matrix",
            f"{prefix}_values_path",
            width=35,
        )
        ttk.Label(frame, text="Constraint kind").grid(row=7, column=0, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=self._var(f"{prefix}_values_kind"),
            values=("equal", "upper", "lower"),
            state="readonly",
            width=10,
        ).grid(row=7, column=1, sticky="w", padx=5)
        ttk.Label(
            frame,
            text="Use NaN for unconstrained entries in value matrices.",
            foreground="#555555",
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=5)
        frame.columnconfigure(1, weight=1)

    def _closure_condition_row(
        self, frame: ttk.LabelFrame, row: int, *, second: bool
    ) -> None:
        token = "closure_second" if second else "closure"
        label = "Second group" if second else "First group"
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
        ttk.Label(frame, text="Kind").grid(row=row, column=1, sticky="e")
        ttk.Combobox(
            frame,
            textvariable=self._var(f"{token}_kind"),
            values=("equality", "least_squares", "lower_equal"),
            state="readonly",
            width=14,
        ).grid(row=row, column=2, padx=5)
        ttk.Label(frame, text="Target").grid(row=row, column=3, sticky="e")
        ttk.Entry(frame, textvariable=self._var(f"{token}_target"), width=16).grid(
            row=row, column=4, padx=5
        )
        ttk.Label(frame, text="Components").grid(row=row, column=5, sticky="e")
        ttk.Entry(frame, textvariable=self._var(f"{token}_components"), width=20).grid(
            row=row, column=6, padx=5, sticky="ew"
        )
        frame.columnconfigure(6, weight=1)

    def _build_advanced_tab(self) -> None:
        sub = ttk.Notebook(self.advanced_tab)
        sub.pack(fill="both", expand=True)
        correlation = ttk.Frame(sub, padding=12)
        multiway = ttk.Frame(sub, padding=12)
        kinetic = ttk.Frame(sub, padding=12)
        sub.add(correlation, text="Correlation")
        sub.add(multiway, text="Multiway / Tucker")
        sub.add(kinetic, text="Kinetic hard models")
        self._build_correlation_panel(correlation)
        self._build_multiway_panel(multiway)
        self._build_kinetic_panel(kinetic)

    def _build_correlation_panel(self, frame: ttk.Frame) -> None:
        ttk.Checkbutton(
            frame,
            text="Enable correlation/calibration constraint",
            variable=self._var("correlation_enabled"),
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        self._source_row(
            frame, 1, "Reference concentrations", "correlation_reference_path"
        )
        ttk.Label(
            frame,
            text="Finite entries are known concentrations; NaN entries are predicted.",
        ).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(frame, text="Component mask").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self._var("correlation_component_mask")).grid(
            row=3, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Label(frame, text="Regression model").grid(row=4, column=0, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=self._var("correlation_model"),
            values=("global", "local"),
            state="readonly",
            width=12,
        ).grid(row=4, column=1, sticky="w", padx=6)
        ttk.Checkbutton(
            frame,
            text="Correct local matrix effects using first experiment as reference",
            variable=self._var("correlation_matrix_effect"),
        ).grid(row=5, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(
            frame,
            text="Use correlation-compatible spectral normalization",
            variable=self._var("correlation_normalize_spectra"),
        ).grid(row=6, column=1, sticky="w", padx=6)
        frame.columnconfigure(1, weight=1)

    def _build_multiway_panel(self, frame: ttk.Frame) -> None:
        trilinear = ttk.LabelFrame(frame, text="Trilinear / quadrilinear", padding=10)
        trilinear.pack(fill="x")
        ttk.Checkbutton(
            trilinear,
            text="Enable multiway constraint",
            variable=self._var("trilinearity_enabled"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(trilinear, text="Direction").grid(row=0, column=1, sticky="e")
        ttk.Combobox(
            trilinear,
            textvariable=self._var("trilinearity_direction"),
            values=("concentration", "spectra", "both"),
            state="readonly",
            width=14,
        ).grid(row=0, column=2, padx=5)
        ttk.Label(trilinear, text="Shape").grid(row=0, column=3, sticky="e")
        ttk.Combobox(
            trilinear,
            textvariable=self._var("trilinearity_shape"),
            values=("1", "2"),
            state="readonly",
            width=4,
        ).grid(row=0, column=4, padx=5)
        ttk.Label(trilinear, text="Component mask").grid(
            row=1, column=0, sticky="w", pady=5
        )
        ttk.Entry(
            trilinear, textvariable=self._var("trilinearity_component_mask")
        ).grid(row=1, column=1, columnspan=4, sticky="ew", padx=5)
        ttk.Label(trilinear, text="Quadrilinear dimensions (3 integers)").grid(
            row=2, column=0, sticky="w"
        )
        ttk.Entry(trilinear, textvariable=self._var("quadrilinear_dimensions")).grid(
            row=2, column=1, columnspan=4, sticky="ew", padx=5
        )
        trilinear.columnconfigure(2, weight=1)

        tucker = ttk.LabelFrame(frame, text="Tucker interactions", padding=10)
        tucker.pack(fill="x", pady=12)
        ttk.Checkbutton(
            tucker,
            text="Enable Tucker constraint",
            variable=self._var("tucker_enabled"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(tucker, text="Number of matrices (0=auto)").grid(
            row=0, column=1, sticky="e"
        )
        ttk.Entry(tucker, textvariable=self._var("tucker_n_matrices"), width=8).grid(
            row=0, column=2, padx=5
        )
        ttk.Label(
            tucker,
            text="One group per component: 0 omits; positive IDs share a group.",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 4))
        ttk.Label(tucker, text="Mode 1 groups").grid(row=2, column=0, sticky="w")
        ttk.Entry(tucker, textvariable=self._var("tucker_mode1_groups")).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=5
        )
        ttk.Label(tucker, text="Mode 3 groups").grid(row=3, column=0, sticky="w")
        ttk.Entry(tucker, textvariable=self._var("tucker_mode3_groups")).grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=5
        )
        tucker.columnconfigure(1, weight=1)

    def _build_kinetic_panel(self, frame: ttk.Frame) -> None:
        ttk.Checkbutton(
            frame,
            text="Enable kinetic hard modeling",
            variable=self._var("kinetic_enabled"),
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text=(
                "Models share rate constants across selected row blocks. ODE-fitted "
                "colored species replace their mapped ALS concentration profiles."
            ),
            wraplength=950,
            justify="left",
        ).pack(fill="x", pady=8)
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True)
        self.kinetic_list = tk.Listbox(body, height=15, exportselection=False)
        self.kinetic_list.pack(side="left", fill="both", expand=True)
        self.kinetic_list.bind("<Double-1>", lambda _event: self._edit_kinetic())
        buttons = ttk.Frame(body)
        buttons.pack(side="left", fill="y", padx=8)
        ttk.Button(buttons, text="Add...", command=self._add_kinetic).pack(fill="x")
        ttk.Button(buttons, text="Edit...", command=self._edit_kinetic).pack(
            fill="x", pady=5
        )
        ttk.Button(buttons, text="Duplicate", command=self._duplicate_kinetic).pack(
            fill="x"
        )
        ttk.Button(buttons, text="Remove", command=self._remove_kinetic).pack(
            fill="x", pady=5
        )

    def _build_results_tab(self) -> None:
        controls = ttk.Frame(self.results_tab)
        controls.pack(fill="x")
        self.run_button = ttk.Button(
            controls, text="Run MCR-ALS", command=self._run_analysis
        )
        self.run_button.pack(side="left")
        self.progress = ttk.Progressbar(controls, mode="indeterminate", length=180)
        self.progress.pack(side="left", padx=10)
        self.status_label = ttk.Label(controls, text="Ready")
        self.status_label.pack(side="left")
        ttk.Button(controls, text="Export NPZ...", command=self._export_npz).pack(
            side="right"
        )
        ttk.Button(controls, text="Export MAT...", command=self._export_mat).pack(
            side="right", padx=6
        )
        ttk.Button(controls, text="Export CSV...", command=self._export_csv).pack(
            side="right"
        )

        panes = ttk.Panedwindow(self.results_tab, orient="horizontal")
        panes.pack(fill="both", expand=True, pady=(8, 0))
        plot_frame = ttk.Frame(panes)
        summary_frame = ttk.LabelFrame(panes, text="Optimization summary", padding=5)
        panes.add(plot_frame, weight=4)
        panes.add(summary_frame, weight=2)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(8, 7), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.summary_text = tk.Text(summary_frame, wrap="word", state="disabled")
        summary_scroll = ttk.Scrollbar(
            summary_frame, orient="vertical", command=self.summary_text.yview
        )
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        self.summary_text.pack(side="left", fill="both", expand=True)
        summary_scroll.pack(side="right", fill="y")

    def capture_state(self) -> GUIState:
        kwargs: dict[str, Any] = {}
        for item in fields(GUIState):
            if item.name == "kinetic_models":
                continue
            default = getattr(self._defaults, item.name)
            value = self._var(item.name).get()
            if isinstance(default, bool):
                kwargs[item.name] = bool(value)
            elif isinstance(default, int):
                try:
                    kwargs[item.name] = int(str(value).strip())
                except ValueError as exc:
                    label = item.name.replace("_", " ")
                    raise ValueError(f"{label} must be an integer") from exc
            elif isinstance(default, float):
                try:
                    kwargs[item.name] = float(str(value).strip())
                except ValueError as exc:
                    label = item.name.replace("_", " ")
                    raise ValueError(f"{label} must be numeric") from exc
            else:
                kwargs[item.name] = str(value).strip()
        kwargs["kinetic_models"] = list(self.kinetic_models)
        return GUIState(**kwargs)

    def apply_state(self, state: GUIState) -> None:
        for item in fields(GUIState):
            if item.name != "kinetic_models":
                self._var(item.name).set(getattr(state, item.name))
        self.kinetic_models = list(state.kinetic_models)
        self._refresh_kinetic_list()

    def _preview_inputs(self) -> None:
        try:
            state = self.capture_state()
            if not state.data_path:
                raise ValueError("select a data matrix first")
            data = load_matrix(state.data_path)
            initial = create_initial_estimate(state, data)
            components = min(initial.shape)
            self.preview_label.configure(
                text=(
                    f"Data {data.shape[0]} x {data.shape[1]} | initial "
                    f"{initial.shape[0]} x {initial.shape[1]} | {components} components"
                )
            )
        except Exception as exc:
            messagebox.showerror("Input preview failed", str(exc), parent=self.root)

    def _save_generated_estimate(self) -> None:
        try:
            state = self.capture_state()
            if state.initial_method == "file":
                raise ValueError("choose SIMPLISMA or EFA to generate an estimate")
            data = load_matrix(state.data_path)
            estimate = create_initial_estimate(state, data)
        except Exception as exc:
            messagebox.showerror("Cannot generate estimate", str(exc), parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".npy",
            filetypes=(("NumPy", "*.npy"), ("CSV", "*.csv"), ("MATLAB", "*.mat")),
        )
        if not path:
            return
        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            np.savetxt(path, estimate, delimiter=",")
        elif suffix == ".mat":
            savemat(path, {"initial_estimate": estimate})
        else:
            np.save(path, estimate)

    def _add_kinetic(self) -> None:
        KineticModelDialog(self.root, KineticModelInput(), self._append_kinetic)

    def _append_kinetic(self, model: KineticModelInput) -> None:
        self.kinetic_models.append(model)
        self._refresh_kinetic_list()
        self.kinetic_list.selection_set(len(self.kinetic_models) - 1)

    def _selected_kinetic(self) -> int | None:
        selection = self.kinetic_list.curselection()
        if not selection:
            messagebox.showinfo(
                "Kinetic models", "Select a model first", parent=self.root
            )
            return None
        return int(selection[0])

    def _edit_kinetic(self) -> None:
        index = self._selected_kinetic()
        if index is None:
            return

        def replace(model: KineticModelInput) -> None:
            self.kinetic_models[index] = model
            self._refresh_kinetic_list()
            self.kinetic_list.selection_set(index)

        KineticModelDialog(self.root, self.kinetic_models[index], replace)

    def _duplicate_kinetic(self) -> None:
        index = self._selected_kinetic()
        if index is None:
            return
        source = self.kinetic_models[index]
        copied = KineticModelInput(
            **{
                item.name: getattr(source, item.name)
                for item in fields(KineticModelInput)
            }
        )
        copied.name = f"{copied.name} copy"
        self._append_kinetic(copied)

    def _remove_kinetic(self) -> None:
        index = self._selected_kinetic()
        if index is not None:
            del self.kinetic_models[index]
            self._refresh_kinetic_list()

    def _refresh_kinetic_list(self) -> None:
        self.kinetic_list.delete(0, "end")
        for index, model in enumerate(self.kinetic_models, start=1):
            self.kinetic_list.insert("end", f"{index}. {model.name}")

    def _run_analysis(self) -> None:
        if self._running:
            return
        try:
            state = self.capture_state()
            if not state.data_path:
                raise ValueError("select a data matrix")
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            return
        self._running = True
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.status_label.configure(text="Running...")
        self.notebook.select(self.results_tab)
        threading.Thread(target=self._solve_worker, args=(state,), daemon=True).start()

    def _solve_worker(self, state: GUIState) -> None:
        try:
            data = load_matrix(state.data_path)
            initial = create_initial_estimate(state, data)
            options = build_options(
                state, (data.shape[0], data.shape[1]), min(initial.shape)
            )
            result = mcr_als(data, initial, options)
            self._events.put(("result", (data, initial, result)))
        except Exception as exc:
            self._events.put(("error", exc))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                self._running = False
                self.progress.stop()
                self.run_button.configure(state="normal")
                if kind == "error":
                    self.status_label.configure(text="Failed")
                    messagebox.showerror(
                        "MCR-ALS failed", str(payload), parent=self.root
                    )
                else:
                    data, initial, result = payload
                    self.result_data = data
                    self.result_initial = initial
                    self.result = result
                    self.status_label.configure(
                        text=(
                            f"{result.status.replace('_', ' ').title()} - "
                            f"iteration {result.best_iteration}"
                        )
                    )
                    self._show_result(result)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_events)

    def _show_result(self, result: MCRALSResult) -> None:
        self.figure.clear()
        ax_c = self.figure.add_subplot(311)
        ax_s = self.figure.add_subplot(312)
        ax_d = self.figure.add_subplot(313)
        ax_c.plot(result.concentrations)
        ax_c.set_title("Concentration profiles")
        ax_c.set_xlabel("Row")
        ax_c.set_ylabel("Concentration")
        ax_s.plot(result.spectra.T)
        ax_s.set_title("Spectral profiles")
        ax_s.set_xlabel("Column")
        ax_s.set_ylabel("Response")
        iterations = np.arange(1, result.history.lack_of_fit_experimental.size + 1)
        ax_d.plot(
            iterations,
            result.history.lack_of_fit_experimental,
            marker="o",
            label="LOF experimental (%)",
        )
        ax_d.plot(
            iterations,
            result.history.r_squared_percent,
            marker=".",
            label="R^2 (%)",
        )
        ax_d.set_title("ALS diagnostics")
        ax_d.set_xlabel("Iteration")
        ax_d.legend(loc="best")
        self.canvas.draw_idle()

        lines = [
            f"Status: {result.status}",
            f"Iterations run: {result.iterations}",
            f"Best iteration: {result.best_iteration}",
            f"Initial estimate mode: {result.initial_estimate_mode}",
            f"PCA lack of fit: {result.pca_lack_of_fit:.8g}%",
            f"ALS lack of fit (PCA): {result.lack_of_fit[0]:.8g}%",
            f"ALS lack of fit (experimental): {result.lack_of_fit[1]:.8g}%",
            f"R^2: {100.0 * result.r_squared:.8g}%",
            "",
            "Component areas (components x row blocks):",
            format_matrix(result.component_areas),
        ]
        if result.weighted_preprocessing is not None:
            weighted = result.weighted_preprocessing
            lines.extend(
                [
                    "",
                    "Weighted MLPCA:",
                    f"  objective: {weighted.objective:.8g}",
                    f"  iterations: {weighted.iterations}",
                    f"  error flag: {weighted.error_flag}",
                ]
            )
        if result.correlation_history:
            lines.extend(
                [
                    "",
                    "Correlation constraint:",
                    f"  fitted iterations: {len(result.correlation_history)}",
                ]
            )
        if result.kinetic_history:
            lines.extend(["", "Kinetic hard models (last iteration):"])
            for index, fit in enumerate(result.kinetic_history[-1], start=1):
                rates = ", ".join(f"{value:.8g}" for value in fit.rate_constants)
                errors = ", ".join(f"{value:.4g}" for value in fit.standard_errors)
                lines.extend(
                    [
                        f"  Model {index} rates: {rates}",
                        f"  Standard errors: {errors}",
                        f"  Sum squared residuals: {fit.sum_squared_residuals:.8g}",
                    ]
                )
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state="disabled")

    def _require_result(self) -> MCRALSResult | None:
        if self.result is None:
            messagebox.showinfo("Export", "Run an analysis first", parent=self.root)
        return self.result

    def _result_dictionary(self, result: MCRALSResult) -> dict[str, Any]:
        output: dict[str, Any] = {
            "concentrations": result.concentrations,
            "spectra": result.spectra,
            "reconstructed_data": result.reconstructed_data,
            "residual_pca": result.residual_pca,
            "residual_experimental": result.residual_experimental,
            "lack_of_fit": result.lack_of_fit,
            "r_squared": result.r_squared,
            "component_areas": result.component_areas,
            "relative_areas": result.relative_areas,
            "iterations": result.iterations,
            "best_iteration": result.best_iteration,
            "status": result.status,
            "pca_reproduced_data": result.pca_reproduced_data,
            "pca_lack_of_fit": result.pca_lack_of_fit,
            "history_lack_of_fit": result.history.lack_of_fit_experimental,
            "history_r_squared_percent": result.history.r_squared_percent,
            "history_sigma_change_percent": result.history.sigma_change_percent,
        }
        if result.kinetic_history:
            for index, fit in enumerate(result.kinetic_history[-1], start=1):
                output[f"kinetic_{index}_rate_constants"] = fit.rate_constants
                output[f"kinetic_{index}_standard_errors"] = fit.standard_errors
                output[f"kinetic_{index}_ssq"] = fit.sum_squared_residuals
        return output

    def _export_npz(self) -> None:
        result = self._require_result()
        if result is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".npz",
            filetypes=(("NumPy archive", "*.npz"),),
        )
        if path:
            np.savez_compressed(path, **self._result_dictionary(result))

    def _export_mat(self) -> None:
        result = self._require_result()
        if result is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".mat",
            filetypes=(("MATLAB MAT", "*.mat"),),
        )
        if path:
            savemat(path, self._result_dictionary(result), do_compression=True)

    def _export_csv(self) -> None:
        result = self._require_result()
        if result is None:
            return
        directory = filedialog.askdirectory(parent=self.root)
        if not directory:
            return
        target = Path(directory)
        matrices = {
            "concentrations.csv": result.concentrations,
            "spectra.csv": result.spectra,
            "reconstructed_data.csv": result.reconstructed_data,
            "residual_experimental.csv": result.residual_experimental,
            "component_areas.csv": result.component_areas,
            "relative_areas.csv": result.relative_areas,
        }
        for filename, matrix in matrices.items():
            np.savetxt(target / filename, matrix, delimiter=",")

    def _save_config(self) -> None:
        try:
            state = self.capture_state()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".json",
            filetypes=(("JSON configuration", "*.json"),),
        )
        if path:
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(state_to_dict(state), stream, indent=2)

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, filetypes=(("JSON configuration", "*.json"),)
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as stream:
                state = state_from_dict(json.load(stream))
            self.apply_state(state)
        except Exception as exc:
            messagebox.showerror(
                "Cannot load configuration", str(exc), parent=self.root
            )

    def _about(self) -> None:
        messagebox.showinfo(
            "About MCR-ALS Toolbox",
            (
                "Python port of the MATLAB MCR-ALS toolbox.\n\n"
                "The desktop interface and Python API use the same float64 numerical "
                "engine and preserve MATLAB constraint and stopping-rule order."
            ),
            parent=self.root,
        )


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    MCRALSGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
