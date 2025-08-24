from code_generator import build_planar_surface_code
from visualize import plot_surface_code_lattice
import stim
import stim
import numpy as np
from mask import mask_generator
from marginalize import calibrate_start_rates, build_mask_once, cfg
from stats import measure_mask_stats, measure_stacked_mask_stats
from inject import run_batched_data_only
from helpers import print_svg, extract_round_template, get_data_and_ancilla_ids_by_parity, make_M_data_local_from_masks

# Generate the rotated surface code circuit:
distance = 5
rounds = 10
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    rounds=rounds,
    distance=distance,
    after_clifford_depolarization=0.0,
    after_reset_flip_probability=0.0

)
mr_indices = [i for i, instr in enumerate(circuit) if instr.name == 'MR']
for mr in mr_indices:
    print(f'mr: {mr}')
    circuit.insert(mr, stim.CircuitInstruction('CORRELATED_ERROR',  [stim.target_x(1), stim.target_y(1)], [0.5]))


# print(circuit.detector_error_model())
# Compile for fast sampling:
sampler = stim.CompiledDetectorSampler(circuit)
print(f'Total qubits: {circuit.num_qubits} \t Total TICKS: {circuit.num_ticks} \t Total detectors:{circuit.num_detectors} \t Total Measurements: {circuit.num_measurements} \t Total observables: {circuit.num_observables}')

# Sample a batch of detection event outcomes:
shots = 2
stbs=circuit.num_detectors //rounds
result = sampler.sample(shots)  # Result has shape (shots, detectors_per_shot)
res=result.reshape((shots, rounds, stbs))
for r in range(rounds):

    for shot in range(shots):
        x_syndromes = res[shot, r, :(stbs//2)]
        z_syndromes = res[shot, r, (stbs//2):]
        print(f"Shot {shot}, Round {r}:")
        print("  X stabilizers:", x_syndromes)
        print("  Z stabilizers:", z_syndromes)

print(circuit)


print(mr_indices)



data_qus,anchs=get_data_and_ancilla_ids_by_parity(circuit, distance)
print(f'\n\nDistance: {distance}\nData qubits ({len(data_qus)}): {data_qus}\nAnchillas ({len(anchs)}): {anchs}\n All of them ({len(data_qus+anchs)}): {data_qus+anchs}')
all_qubits=data_qus+anchs

###!!!!!!!!!!!!!!!!!!Dont forget to map data qubits to 0-d²-1 !!!!!!!!! 


batch_marg=256
iters_marg=15

cfg_ = calibrate_start_rates(
    build_mask_once=build_mask_once,
    qubits=len(data_qus), 
    rounds=rounds, 
    qubits_ind=data_qus,
    cfg=cfg,
    p_idle_target=cfg["p_idle"],
    batch=batch_marg,
    iters=iters_marg,
    tol=0.05,   # ±15% is plenty for training
    verbose=True
)
print(f"\n\nCalibrated start_probs (batch: {batch_marg}, iters: {iters_marg}):")
for key in ("t1", "t2", "t3", "t4"):
    if key in cfg_ and cfg_[key].get("enabled", False) and "p_start" in cfg_[key]:
        print(cfg_[key]["p_start"])



## I am thinking of generating M_data and M_anchilla and M_CNOT. this way, we will have them ckeared out, so somehow we can combine them at the end 
cat_counts=[0, 0, 0, 0]

for _ in range(1000):
    #use the calibrated cfg to generate masks
    M_data, actives=mask_generator(qubits=len(data_qus), rounds=rounds, qubits_ind=data_qus, cfg=cfg_, actives_list=True)
    #measure_mask_stats(m=M_data, actives=actives)
    cat_counts=[c+int(b) for c,b in zip(cat_counts, actives)]
#print(f'\nFinal Mask M0:\n{M_data}')
print(f'Each category coutns: {cat_counts}')


M_data=mask_generator(qubits=len(data_qus), rounds=rounds, qubits_ind=data_qus, cfg=cfg_, actives_list=False)
print(f'\nFinal Mask M0:\n{M_data}\n\n\n')


#---------------------------------  FlipSIm-------------------------------------------------

S=1024
res = [
    mask_generator(
        qubits=len(data_qus),
        rounds=rounds,
        qubits_ind=data_qus,   # you’re already generating data-only rows
        cfg=cfg_,
        actives_list=True
    )
    for _ in range(S)
]

# Unpack results
Ms, cats = zip(*res)                 # Ms: tuple of (D,R) arrays; cats: tuple of [c1,c2,c3,c4]
M_data = np.stack(Ms, axis=-1).astype(np.int8)       # -> shape (D, R, S)
actives = np.array(cats, dtype=np.int32).T           # -> shape (4, S)

print("M_data shape:", M_data.shape)
print("actives shape:", actives.shape)


M_data_local=make_M_data_local_from_masks(masks=M_data, data_ids=data_qus, rounds=rounds )
print(f'M local is of shape: {M_data_local.shape}')



prefix, pre_round, meas_round, anc_ids, repeat_count = extract_round_template(circuit)


circ_by_round = [(pre_round, stim.Circuit(), meas_round) for _ in range(rounds)]

round0 = stim.Circuit(); round0 += pre_round; round0 += meas_round
print_svg(round0, "r0")
print_svg(circuit, "circuit")
#print(round0)   #only one form #round 


# for i, item1 in enumerate(circ_by_round):
#     print(f'round?{i}:\n ')
#     for j in circ_by_round[i]:

#         print(f'item :\n{j}\n')



print(f'Lets check ids:\nData: {data_qus}\nAnchillas: {anchs}')  #  (All ok!)

stats = measure_stacked_mask_stats(M_data_local, "M_data_local")
# sanity vs target p_idle:
p_idle = 0.005
print("\n\ntarget p_idle =", p_idle, "  measured p̂ =", stats["p_hat"])


##Check wtf these are and the physics behind them 
cnt = stats["per_shot_counts"]            # from the meter we made
print("mean", cnt.mean(), "var", cnt.var(), "Fano", cnt.var()/max(1e-9,cnt.mean()))
# Fano >> 1 means bursty (correlated); Fano ~ 1 means near-Poisson iid.

## einai entaksei gia twra na xrhsimopoimv chat giati kanw ta statistika klp. Sto montelo kane kai tipota moni sou.
#  Den einai na paizoyme me ton kwdika gia ta statistika, marginalization klp



names = ("spatial", "temporal", "cluster_ext", "multi_scattered")

print(f'\n\n')
for i, name in enumerate(names):
    shots_with = int((actives[i] > 0).sum())
    total = int(actives[i].sum())
    print(f"{name}: shots_with≥1={shots_with}/{S}  total_events={total}")
print(f'\n\n')


dets, obs, meas = run_batched_data_only(circuit=circuit,M_data= M_data_local, data_ids=data_qus)
print('Some STATS:\n')
print(f'Detector Flips: \n{dets.any()}\n\nObservations: \n{obs}\n\n Measurement Flips: \n{meas.any()}\n')




print("\t*detector flip rate:", dets.mean())              # fraction of (detector,shot) that are True
print("\t *measurement flip rate:", meas.mean())           # fraction of (measurement,shot) that are True
print("\t *shots with any detector flip:", (dets.any(axis=0)).mean())
print("\t *detectors that fired at least Once:", int(dets.any(axis=1).sum()), "/", dets.shape[0])


# How many shots each detector fired in:
det_counts = dets.sum(axis=1)              # int counts per detector
print("\t *top-5 detectors by count:", det_counts[:5])

# Which shots had any detector flip?
shots_with = dets.any(axis=0)              # shape (S,)
print("\t *shots with any detector flip:", shots_with.sum(), "/", shots_with.size)


# # force Z on first data row in round 0 for first 16 shots
# M_forced = M_data_local.copy()
# M_forced[:] = 0
# M_forced[0, 0, :16] = 2  # your codes: X=1, Z=2, Y=3
# dets_f, _, _ = run_batched_data_only(circuit, M_forced, data_ids=data_qus)

# who = np.where(dets_f.any(axis=1))[0]
# print("detectors triggered by forced Z (subset):", who[:20])




# Toy: ancilla a=0, data d=1. Measure an X-check on data via H-CNOT-H then MR(a).
# toy = stim.Circuit("""
# H 0
# CX 0 1
# H 0
# MR 0
# DETECTOR rec[-1]
# """)

# S = 16
# sim = stim.FlipSimulator(batch_size=S, num_qubits=2, disable_stabilizer_randomization=True)

# # Inject a Z on data (q=1) BEFORE the round
# xmask = np.zeros((2, S), np.bool_)
# ymask = np.zeros((2, S), np.bool_)
# zmask = np.zeros((2, S), np.bool_)
# zmask[1, :] = True

# sim.broadcast_pauli_errors(pauli='Z', mask=zmask)  # always apply
# sim.do(toy)

# dets = sim.get_detector_flips()
# print("toy dets any?", dets.any())   # EXPECT: True
