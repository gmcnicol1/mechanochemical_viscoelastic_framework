# Mechanochemical PDE Framework: method-of-Lines implementation with state-dependent mass matrices

This repository contains numerical implementations of a general mechanochemical modelling framework based on the method of lines (MOL), sparse finite-difference discretisation, and adaptive implicit time integration. The framework is designed for coupled systems containing biochemical reaction-diffusion-advection processes and viscoelastic mechanical deformation, where constitutive properties may depend dynamically on the evolving biochemical state.

Two independent implementations are provided:

1. Minimal mechanochemical model
   - A reduced reaction-diffusion–mechanics system demonstrating the general computational framework.
   - Includes multiple viscoelastic constitutive representations:
     - Kelvin–Voigt
     - Maxwell
     - Standard Linear Solid (Kelvin representation)
     - Standard Linear Solid (Maxwell representation)

2. Cell–substrate interaction model
   - A larger mechanochemical model describing coupled cellular signalling, adhesion dynamics, cytoskeletal organisation, and substrate deformation.
   - Demonstrates application of the framework to a biologically motivated PDE system with strong biochemical–mechanical feedback.

The two implementations use separate driver scripts and model-specific computational modules. However, both follow the same underlying numerical framework based on finite-difference spatial discretisation, sparse mass-matrix formulation, and adaptive stiff ODE integration.

---

# Repository structure

├── README.md
│
├── minimal_example
│   ├── driver_code.py
│   ├── overall_differential_equations.py
│   ├── biochemistry_differential_equations.py
│   ├── KV_differential_equations.py
│   ├── Maxwell_differential_equations.py
│   ├── SLS_Kelvin_differential_equations.py
│   ├── SLS_Maxwell_differential_equations.py
│   └── postprocessing codes
│
└── cell_substrate_model
    ├── driver_code.py
    ├── cell_ECM_differential_equations.py
    ├── signalling_proteins_ODEs.py
    ├── scaffolding_proteins_ODEs.py
    ├── mechanics_ODEs.py
    └── postprocessing codes

---

# Requirements

The code is implemented in Python 3.13.

Required packages:

- NumPy
- SciPy
- Matplotlib

---

# Numerical framework

The governing equations are formulated in the general operator form

\mathcal{M}(y,x,t)\frac{\partial y}{\partial t}=\mathcal{F}(y,x,t),

where the temporal operator may contain both local contributions and spatial operators acting on time derivatives.

Following spatial discretisation using second-order finite differences, the continuous system becomes the semi-discrete mass-matrix formulation

\mathbf{M}(y)\frac{d\mathbf{y}}{dt}=\mathbf{F}(y,t).

The resulting systems are stiff due to the presence of:

- biochemical diffusion and transport,
- viscoelastic relaxation,
- reaction kinetics,
- nonlinear biochemical–mechanical feedback.

Consequently, adaptive implicit time integration methods are used for all simulations.

The finite-difference discretisation produces sparse mass matrices because spatial operators only couple neighbouring grid points. Sparse matrix storage and solution techniques are therefore employed to improve computational efficiency for increasing spatial resolution.

---

# Minimal mechanochemical model

The minimal example provides a reduced mechanochemical system for demonstrating the general framework and verifying incorporation of different constitutive models within the same MOL formulation.

The model contains:

- two reacting biochemical species,
- one mechanical displacement field,
- biochemical regulation of mechanical properties,
- deformation-dependent biochemical feedback.

The implemented rheological descriptions are:

- Kelvin–Voigt material,
- Maxwell material,
- Standard Linear Solid represented in Kelvin form,
- Standard Linear Solid represented in Maxwell form.

Each constitutive model is implemented independently through its own mechanical module.

## Running the minimal model

Navigate to the minimal model directory: minimal_example

Run the driver: python driver_code.py

The selected mechanical formulation can be specified through the corresponding rheological implementation:
KV_differential_equations.py
Maxwell_differential_equations.py
SLS_Kelvin_differential_equations.py
SLS_Maxwell_differential_equations.py

The workflow is:

1. Initialise the spatial domain, parameters, and initial conditions.
2. Evaluate biochemical reaction-diffusion dynamics. 
3. Construct the selected mechanical operator and sparse mass matrix.
4. Assemble the coupled right-hand-side vector.
5. Advance the stiff system using an adaptive implicit solver.
6. Store solution outputs for post-processing and visualisation.

---

# Cell–substrate interaction model

The cell–substrate model demonstrates application of the framework to a larger mechanochemical system involving coupled intracellular signalling, adhesion dynamics, cytoskeletal organisation, and substrate deformation.

The model contains:

- biochemical reaction-diffusion-advection equations describing cellular signalling and adhesion processes,
- coupled viscoelastic deformation equations for the cell and extracellular matrix.

The fully discretised system contains 24N+28 state variables.

## Running the cell–substrate model

Navigate to the cell-substrate model directory: cell_substrate_model
Run the driver: driver_code.py
The coupled system is evaluated through: cell_ECM_differential_equations.py which calls separate computational modules for the biological subsystems:

### Signalling dynamics

Implemented in: signalling_proteins_ODEs.py

This module computes reaction-diffusion dynamics associated with intracellular signalling pathways.

### Adhesion and cytoskeletal dynamics

Implemented in: scaffolding_proteins_ODEs.py

This module computes dynamics associated with:

- integrin activation,
- adhesion formation,
- actin polymerisation,
- myosin activation,
- stress fibre assembly.

### Mechanical deformation

Implemented in: mechanics_ODEs.py

This module computes coupled deformation of the cell and extracellular matrix.

The mechanical subsystem internally assembles and solves the structured sparse state-dependent mass matrix generated by the viscoelastic constitutive formulation.

---

# Spatial discretisation and boundary conditions

All models use:

- uniform spatial grids,
- second-order finite differences,
- method-of-lines reduction,
- sparse mass-matrix formulation.

Biochemical ghost nodes are eliminated algebraically where possible using the imposed boundary conditions.

Mechanical boundary conditions depend on the specific model.

For the minimal model, prescribed displacement conditions are used: u(0,t)=0, u(1,t)=f_R(t).
For the cell–substrate model, stress-free mechanical boundaries are implemented using fictitious nodes retained explicitly in the discretised system.

---

# Output and post-processing

Simulation outputs include:

- biochemical concentration profiles,
- displacement fields,
- stress and strain distributions,
- adhesion dynamics,
- cytoskeletal organisation,
- mechanochemical feedback responses.

Post-processing scripts generate spatial profiles and temporal trajectories for analysis and visualisation.

For the minimal example, post-processing scripts are example_graphics_paper.py and comparative_graphics_paper.py; for the cell-substrate interaction example, post-processing scripts are full_example_graphics_paper.py and feedback_graphics_paper.py

---

# Extending the framework

The modular structure allows extension through:

- additional biochemical species,
- alternative reaction networks,
- new constitutive models,
- alternative mechanical boundary conditions,
- additional biochemical–mechanical coupling mechanisms.

New mechanical formulations can be incorporated by implementing:

1. The constitutive relation.
2. The corresponding temporal operator.
3. The forcing operator.
4. The sparse mass-matrix assembly routine.

---

# Reproducibility

All simulations associated with the accompanying manuscript can be reproduced directly from the provided driver scripts.

To reproduce results:

1. Navigate to the required model directory.
2. Run the corresponding `driver_code.py`.
3. Use the provided post-processing scripts to generate figures and analyse outputs.