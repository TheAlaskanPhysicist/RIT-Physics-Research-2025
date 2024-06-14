
Introduction to the Einstein Toolkit (Left off at 24:13)
https://www.youtube.com/watch?v=bJgE_Jb1dvs


## Summary
### Goals
 - Community driven
 - Core computational tool for numerical astrophysics
 - General purpose tool
### Components
 - Cactus
 - Simulation Factory
 - Kranc
 - NRPy+
 - Science Modules






## Einstein Toolkit's Purpose
**Addresses**
Contributing with many people for increased complexity
More diverse hardware (Intel/AMD x64, GPUs, new ARM processors)

### Computational Challenges
 - Simulate cutting-edge science
 - Use latest numerical methods
 - Make use of latest hardware 
	 - Vector Ops      (Krank, NRPy+)
	 - Core Scaling   (OpenMP)                      |InterCPU|
	 - Node Scaling (MPI, Carpet, CarpetX)    |IntraCPU|
	 - AMR (Adaptive Mesh Refinement, Carpet, CarpetX, MoL)
	 - GPU Acceleration (CarpetX)
	 - Machine Learning? (new study)

### Mundane Challenges
 - Efficient I/O
 - HDF5
 - Checkpoint/Restart
 - Parameter Parsing
 - Visualization
 - Analysis

### Collaborative Challenges
 - Building infrastructure for many to use.
 - Einstein Toolkit has standards to keep things consistent.
 - Using C/C++, C, Perl/C, Fortran, etc.
 - Accreditation (making sure the people get credit).

### Domain Decomposition (example)
Without Ghost-Zones:
	Couldn't do processor overlap (insufficient data for calculation)
With Ghost-Zones:
	Give processor a copy of the data adjacent for calculations.

This is identical to what I learned from Dr. Zlochower.

### Multiblock and Refinement (example)
Different regions can have different coordinate systems and allowing for decomp on it becomes significantly harder.









