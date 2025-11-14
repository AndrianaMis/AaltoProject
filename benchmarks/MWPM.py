import stim
import numpy as np
import pymatching

# Assuming your noise model structure is defined outside this function
# (e.g., in your M0, M1, M2 generators or in a global noise_params dictionary)

def generate_dem_and_decoder(surface_code_circuit: stim.Circuit,
                             p_physical: float,
                             distance: int):
    """
    Generates the Detector Error Model (DEM) and initializes the MWPM decoder.
    """
    print(f"Generating Detector Error Model for d={distance}...")
    
    # 1. Generate the DEM from the full Stim circuit.
    # This automatically includes the geometry and connectivity.
    # Note: Stim's `detector_error_model` attempts to marginalize over the noise.
    # The complexity of M1/M2 noise may require specific sampling depending on the
    # correlation length, but for a standard comparison, this is the first step.
    dem = surface_code_circuit.detector_error_model(
        approximate_disjoint_errors_after_rounds=9 # Use your 9 REPEAT rounds
    )
    
    # 2. Convert the DEM to a pymatching.Matching object.
    # The matching graph is the core of the MWPM algorithm.
    mwpm_decoder = pymatching.Matching.from_detector_error_model(dem)
    
    return mwpm_decoder



