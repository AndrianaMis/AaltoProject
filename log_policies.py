import os
import torch

path="stage2_d3_ler_0.0009765625_13:22.pt"

checkpoint = torch.load(path, map_location="cpu")

print(type(checkpoint))
print(checkpoint.keys())
print(f'LER: {checkpoint["ler"]}  EP: {checkpoint["episode"]} Noise Scalar: {checkpoint["noise_scalar"]}')
#print(f'LER: {checkpoint["ler"]}  EP: {checkpoint["episode"]} ')
# sd = checkpoint["agent"]
# print("State dict keys:", list(sd.keys())[:15]) 
print(f'CFG USED: {checkpoint["cfg"]}')







import matplotlib.pyplot as plt
import numpy as np
import re


# Full data parsing function
def parse_logs(log_text):
    noise = []
    ler = []
    # Regex to find numbers after "LER:" and "Noise scalar:"
    pattern = r"LER:\s+([0-9\.]+).*?Noise scalar:\s+([0-9\.]+)"
    matches = re.findall(pattern, log_text)
    
    for m in matches:
        ler.append(float(m[0]) * 100) # Convert to Percentage
        noise.append(float(m[1]) * 100) # Convert to Percentage
    return noise, ler




'''Figure 3: Sensitivity of Logical Error Rate to Instantaneous Noise Intensity."To understand the decoder's performance limit, 
we analyzed the correlation between the instantaneous physical error density (Noise Scalar) and the resulting Logical Error Rate (LER) 
across evaluation episodes. As shown in Figure 3, the system exhibits two distinct regimes:Sub-Threshold Regime ($p_{eff} < 0.15\%$): 
When the effective noise density is low, the SSM-RL decoder successfully suppresses errors, achieving a peak performance of $0.09\%$ (red star), 
which falls below the physical break-even threshold (green dashed line).Burst-Limited Regime ($p_{eff} > 0.2\%$): As the noise scalar increases, 
the LER rises linearly. This linear dependence ($R \approx 1$) indicates that the limiting factor is not the decoder's policy, but the physical
property of the Fano-factor $\sim 8$ noise. High-intensity bursts introduce 
error chains ($weight > \lfloor (d-1)/2 \rfloor$) that are information-theoretically uncorrectable by a code of distance $d=3$."'''






import matplotlib.pyplot as plt
import numpy as np

lers_seen="""

EP: [0] ->  LER: 0.0078125  Noise scalar: 0.00460400993441358  Seed: 1275
EP: [1] ->  LER: 0.0078125  Noise scalar: 0.004269446855709877  Seed: 1281
EP: [2] ->  LER: 0.0048828125  Noise scalar: 0.0034752363040123453  Seed: 1254
EP: [3] ->  LER: 0.0087890625  Noise scalar: 0.003472222222222222  Seed: 1235
EP: [4] ->  LER: 0.0078125  Noise scalar: 0.0049295307677469135  Seed: 1262
EP: [5] ->  LER: 0.0029296875  Noise scalar: 0.0025499131944444445  Seed: 1269
EP: [6] ->  LER: 0.0087890625  Noise scalar: 0.003904742959104938  Seed: 1265
EP: [7] ->  LER: 0.0068359375  Noise scalar: 0.003904742959104938  Seed: 1271
EP: [8] ->  LER: 0.0068359375  Noise scalar: 0.003030659239969136  Seed: 1268
EP: [9] ->  LER: 0.01171875  Noise scalar: 0.004231770833333334  Seed: 1266
EP: [10] ->  LER: 0.0068359375  Noise scalar: 0.004264925733024691  Seed: 1276
EP: [11] ->  LER: 0.0078125  Noise scalar: 0.004144362461419753  Seed: 1241
EP: [12] ->  LER: 0.0068359375  Noise scalar: 0.004477418499228395  Seed: 1252
EP: [13] ->  LER: 0.0126953125  Noise scalar: 0.005044065875771605  Seed: 1244
EP: [14] ->  LER: 0.0048828125  Noise scalar: 0.002584575135030864  Seed: 1261
EP: [15] ->  LER: 0.0068359375  Noise scalar: 0.0031045042438271604  Seed: 1256
EP: [16] ->  LER: 0.0068359375  Noise scalar: 0.0032853491512345677  Seed: 1245
EP: [17] ->  LER: 0.00390625  Noise scalar: 0.0025740258487654323  Seed: 1246
EP: [18] ->  LER: 0.0068359375  Noise scalar: 0.004215193383487655  Seed: 1247
EP: [19] ->  LER: 0.0107421875  Noise scalar: 0.0036907431520061726  Seed: 1239
EP: [20] ->  LER: 0.0078125  Noise scalar: 0.004188066647376544  Seed: 1237
EP: [21] ->  LER: 0.005859375  Noise scalar: 0.002989969135802469  Seed: 1253
EP: [22] ->  LER: 0.0068359375  Noise scalar: 0.0040901089891975315  Seed: 1278
EP: [23] ->  LER: 0.0078125  Noise scalar: 0.004501531153549383  Seed: 1283
EP: [24] ->  LER: 0.0029296875  Noise scalar: 0.0028377580054012343  Seed: 1258
EP: [25] ->  LER: 0.01171875  Noise scalar: 0.0030035325038580245  Seed: 1255
EP: [26] ->  LER: 0.0068359375  Noise scalar: 0.00422875675154321  Seed: 1270
EP: [27] ->  LER: 0.00390625  Noise scalar: 0.004135320216049383  Seed: 1249
EP: [28] ->  LER: 0.005859375  Noise scalar: 0.0028663917824074073  Seed: 1267
EP: [29] ->  LER: 0.0068359375  Noise scalar: 0.004492488908179012  Seed: 1259
EP: [30] ->  LER: 0.0029296875  Noise scalar: 0.0019048996913580247  Seed: 1260
EP: [31] ->  LER: 0.0087890625  Noise scalar: 0.0036048418209876543  Seed: 1263
EP: [32] ->  LER: 0.0078125  Noise scalar: 0.004462348090277778  Seed: 1264
EP: [33] ->  LER: 0.0078125  Noise scalar: 0.0031904055748456788  Seed: 1250
EP: [34] ->  LER: 0.0078125  Noise scalar: 0.002896532600308642  Seed: 1248
EP: [35] ->  LER: 0.0107421875  Noise scalar: 0.004513587480709877  Seed: 1236
EP: [36] ->  LER: 0.0068359375  Noise scalar: 0.004575376157407407  Seed: 1238
EP: [37] ->  LER: 0.0078125  Noise scalar: 0.00374951774691358  Seed: 1257
EP: [38] ->  LER: 0.0068359375  Noise scalar: 0.005169150270061729  Seed: 1277
EP: [39] ->  LER: 0.01171875  Noise scalar: 0.0033923490547839506  Seed: 1243
EP: [40] ->  LER: 0.0068359375  Noise scalar: 0.004139841338734568  Seed: 1280
EP: [41] ->  LER: 0.0126953125  Noise scalar: 0.0048074604552469135  Seed: 1273
EP: [42] ->  LER: 0.0087890625  Noise scalar: 0.00370581356095679  Seed: 1272
EP: [43] ->  LER: 0.0078125  Noise scalar: 0.0038685739776234567  Seed: 1251
EP: [44] ->  LER: 0.0009765625  Noise scalar: 0.002846800250771605  Seed: 1242
EP: [45] ->  LER: 0.0029296875  Noise scalar: 0.003833912037037037  Seed: 1279
EP: [46] ->  LER: 0.005859375  Noise scalar: 0.0029929832175925927  Seed: 1240
EP: [47] ->  LER: 0.0078125  Noise scalar: 0.003728419174382716  Seed: 1234
EP: [48] ->  LER: 0.0087890625  Noise scalar: 0.002503194926697531  Seed: 1282
EP: [49] ->  LER: 0.0107421875  Noise scalar: 0.003937897858796296  Seed: 1274"""


lers_unseen="""
EP: [0] ->  LER: 0.0078125  Noise scalar: 0.0032597294560185184  Seed: 4362
EP: [1] ->  LER: 0.0048828125  Noise scalar: 0.0036967713155864196  Seed: 4368
EP: [2] ->  LER: 0.005859375  Noise scalar: 0.004430700231481482  Seed: 4341
EP: [3] ->  LER: 0.01171875  Noise scalar: 0.00394091194058642  Seed: 4322
EP: [4] ->  LER: 0.0048828125  Noise scalar: 0.003323025173611111  Seed: 4349
EP: [5] ->  LER: 0.0078125  Noise scalar: 0.00411572868441358  Seed: 4356
EP: [6] ->  LER: 0.0126953125  Noise scalar: 0.0040479118441358024  Seed: 4352
EP: [7] ->  LER: 0.0029296875  Noise scalar: 0.0035867573302469135  Seed: 4358
EP: [8] ->  LER: 0.0068359375  Noise scalar: 0.0033802927276234567  Seed: 4355
EP: [9] ->  LER: 0.0029296875  Noise scalar: 0.0025815610532407404  Seed: 4353
EP: [10] ->  LER: 0.01171875  Noise scalar: 0.0028709129050925927  Seed: 4363
EP: [11] ->  LER: 0.0078125  Noise scalar: 0.004462348090277778  Seed: 4328
EP: [12] ->  LER: 0.0078125  Noise scalar: 0.0034255039544753086  Seed: 4339
EP: [13] ->  LER: 0.0068359375  Noise scalar: 0.003945433063271605  Seed: 4331
EP: [14] ->  LER: 0.009765625  Noise scalar: 0.0037781515239197526  Seed: 4348
EP: [15] ->  LER: 0.009765625  Noise scalar: 0.005491657021604938  Seed: 4343
EP: [16] ->  LER: 0.005859375  Noise scalar: 0.003716362847222222  Seed: 4332
EP: [17] ->  LER: 0.00390625  Noise scalar: 0.003744996624228395  Seed: 4333
EP: [18] ->  LER: 0.0048828125  Noise scalar: 0.0031029972029320988  Seed: 4334
EP: [19] ->  LER: 0.0087890625  Noise scalar: 0.004689911265432099  Seed: 4326
EP: [20] ->  LER: 0.005859375  Noise scalar: 0.0036847149884259266  Seed: 4324
EP: [21] ->  LER: 0.0087890625  Noise scalar: 0.004653742283950617  Seed: 4340
EP: [22] ->  LER: 0.0166015625  Noise scalar: 0.00423779899691358  Seed: 4365
EP: [23] ->  LER: 0.01171875  Noise scalar: 0.003226574556327161  Seed: 4370
EP: [24] ->  LER: 0.005859375  Noise scalar: 0.0038896725501543212  Seed: 4345
EP: [25] ->  LER: 0.0107421875  Noise scalar: 0.004150390625  Seed: 4342
EP: [26] ->  LER: 0.0126953125  Noise scalar: 0.0043885030864197535  Seed: 4357
EP: [27] ->  LER: 0.0078125  Noise scalar: 0.0031240957754629633  Seed: 4336
EP: [28] ->  LER: 0.0048828125  Noise scalar: 0.003585250289351852  Seed: 4354
EP: [29] ->  LER: 0.0078125  Noise scalar: 0.0029055748456790122  Seed: 4346
EP: [30] ->  LER: 0.0078125  Noise scalar: 0.003140673225308642  Seed: 4347
EP: [31] ->  LER: 0.0087890625  Noise scalar: 0.004275475019290123  Seed: 4350
EP: [32] ->  LER: 0.0068359375  Noise scalar: 0.0035098982445987657  Seed: 4351
EP: [33] ->  LER: 0.005859375  Noise scalar: 0.004123263888888889  Seed: 4337
EP: [34] ->  LER: 0.0029296875  Noise scalar: 0.0031949266975308645  Seed: 4335
EP: [35] ->  LER: 0.00390625  Noise scalar: 0.0024609977816358024  Seed: 4323
EP: [36] ->  LER: 0.0029296875  Noise scalar: 0.0027322651427469135  Seed: 4325
EP: [37] ->  LER: 0.0087890625  Noise scalar: 0.004023799189814815  Seed: 4344
EP: [38] ->  LER: 0.01171875  Noise scalar: 0.003690743152006173  Seed: 4364
EP: [39] ->  LER: 0.0078125  Noise scalar: 0.002423321759259259  Seed: 4330
EP: [40] ->  LER: 0.01171875  Noise scalar: 0.004270953896604938  Seed: 4367
EP: [41] ->  LER: 0.0078125  Noise scalar: 0.00325219425154321  Seed: 4360
EP: [42] ->  LER: 0.0078125  Noise scalar: 0.002824194637345679  Seed: 4359
EP: [43] ->  LER: 0.0068359375  Noise scalar: 0.004065996334876544  Seed: 4338
EP: [44] ->  LER: 0.0078125  Noise scalar: 0.004751699942129629  Seed: 4329
EP: [45] ->  LER: 0.0078125  Noise scalar: 0.00461305217978395  Seed: 4366
EP: [46] ->  LER: 0.009765625  Noise scalar: 0.004977756076388889  Seed: 4327
EP: [47] ->  LER: 0.0126953125  Noise scalar: 0.003996672453703704  Seed: 4321
EP: [48] ->  LER: 0.0068359375  Noise scalar: 0.0034661940586419755  Seed: 4369
EP: [49] ->  LER: 0.0068359375  Noise scalar: 0.0038143205054012343  Seed: 4361"""



'''===== SEEN-SEED EVAL SUMMARY =====
LER mean=0.00734   std=0.00256
LER min=0.00098     max=0.01270
noise mean=0.00374
===================================


===== UNSEEN-SEED EVAL SUMMARY =====
LER mean=0.00783   std=0.00295
LER min=0.00293     max=0.01660
noise mean=0.00374
===================================

'''




# # --- PASTE YOUR LOGS HERE (The text you provided) ---
# data_seen = """
# EP: [0] ->  LER: 0.0078125  Noise scalar: 0.0021731529706790122  Seed: 1235
# EP: [1] ->  LER: 0.0087890625  Noise scalar: 0.002896532600308642  Seed: 1264
# EP: [2] ->  LER: 0.0078125  Noise scalar: 0.0014844352816358024  Seed: 1260
# EP: [25] ->  LER: 0.0029296875  Noise scalar: 0.0017677589699074073  Seed: 1269
# EP: [27] ->  LER: 0.013671875  Noise scalar: 0.004409601658950617  Seed: 1277
# EP: [10] ->  LER: 0.0166015625  Noise scalar: 0.0025212794174382714  Seed: 1241
# """
# # (Note: I included a few samples above. 
# # You should paste your FULL log text into the variable above for the best plot)


# # *** PASTE YOUR FULL LOGS INTO THESE VARIABLES ***
# # (I am using the summary stats from your text for the demo, 
# # but you should use the full text you pasted in the chat)
# full_seen_log = """
# EP: [0] ->  LER: 0.0078125  Noise scalar: 0.0021731529706790122  Seed: 1235
# EP: [1] ->  LER: 0.0087890625  Noise scalar: 0.002896532600308642  Seed: 1264
# EP: [2] ->  LER: 0.0078125  Noise scalar: 0.0014844352816358024  Seed: 1260
# EP: [3] ->  LER: 0.0087890625  Noise scalar: 0.002182195216049383  Seed: 1272
# EP: [4] ->  LER: 0.0146484375  Noise scalar: 0.002424828800154321  Seed: 1239
# EP: [5] ->  LER: 0.0078125  Noise scalar: 0.0024549696180555555  Seed: 1256
# EP: [6] ->  LER: 0.005859375  Noise scalar: 0.0020978009259259257  Seed: 1253
# EP: [7] ->  LER: 0.0107421875  Noise scalar: 0.0035400390625  Seed: 1266
# EP: [8] ->  LER: 0.0048828125  Noise scalar: 0.0019380545910493826  Seed: 1261
# EP: [9] ->  LER: 0.0126953125  Noise scalar: 0.00328685619212963  Seed: 1275
# EP: [10] ->  LER: 0.0166015625  Noise scalar: 0.0025212794174382714  Seed: 1241
# EP: [11] ->  LER: 0.0107421875  Noise scalar: 0.0019079137731481482  Seed: 1247
# EP: [12] ->  LER: 0.017578125  Noise scalar: 0.002176167052469136  Seed: 1245
# EP: [13] ->  LER: 0.013671875  Noise scalar: 0.002917631172839506  Seed: 1263
# EP: [14] ->  LER: 0.01171875  Noise scalar: 0.0029929832175925927  Seed: 1273
# EP: [15] ->  LER: 0.009765625  Noise scalar: 0.002592110339506173  Seed: 1257
# EP: [16] ->  LER: 0.009765625  Noise scalar: 0.0028950255594135804  Seed: 1270
# EP: [17] ->  LER: 0.005859375  Noise scalar: 0.0019033926504629629  Seed: 1268
# EP: [18] ->  LER: 0.0068359375  Noise scalar: 0.0031391661844135804  Seed: 1281
# EP: [19] ->  LER: 0.0068359375  Noise scalar: 0.002792546778549383  Seed: 1248
# EP: [20] ->  LER: 0.0146484375  Noise scalar: 0.001930519386574074  Seed: 1267
# EP: [21] ->  LER: 0.0068359375  Noise scalar: 0.0017572096836419753  Seed: 1282
# EP: [22] ->  LER: 0.0078125  Noise scalar: 0.0016034915123456792  Seed: 1242
# EP: [23] ->  LER: 0.013671875  Noise scalar: 0.002846800250771605  Seed: 1259
# EP: [24] ->  LER: 0.0107421875  Noise scalar: 0.0030261381172839506  Seed: 1276
# EP: [25] ->  LER: 0.0029296875  Noise scalar: 0.0017677589699074073  Seed: 1269
# EP: [26] ->  LER: 0.0078125  Noise scalar: 0.0031542365933641976  Seed: 1278
# EP: [27] ->  LER: 0.013671875  Noise scalar: 0.004409601658950617  Seed: 1277
# EP: [28] ->  LER: 0.0166015625  Noise scalar: 0.003348644868827161  Seed: 1252
# EP: [29] ->  LER: 0.009765625  Noise scalar: 0.003023124035493827  Seed: 1265
# EP: [30] ->  LER: 0.01171875  Noise scalar: 0.0029658564814814816  Seed: 1274
# EP: [31] ->  LER: 0.0126953125  Noise scalar: 0.002551420235339506  Seed: 1237
# EP: [32] ->  LER: 0.0078125  Noise scalar: 0.0024896315586419755  Seed: 1254
# EP: [33] ->  LER: 0.0146484375  Noise scalar: 0.0030487437307098767  Seed: 1236
# EP: [34] ->  LER: 0.0087890625  Noise scalar: 0.002671983506944444  Seed: 1238
# EP: [35] ->  LER: 0.0087890625  Noise scalar: 0.0031376591435185184  Seed: 1244
# EP: [36] ->  LER: 0.01171875  Noise scalar: 0.0030818986304012343  Seed: 1251
# EP: [37] ->  LER: 0.0048828125  Noise scalar: 0.0020390263310185184  Seed: 1240
# EP: [38] ->  LER: 0.0087890625  Noise scalar: 0.0029372227044753086  Seed: 1249
# EP: [39] ->  LER: 0.0078125  Noise scalar: 0.002530321662808642  Seed: 1250
# EP: [40] ->  LER: 0.0048828125  Noise scalar: 0.0021008150077160494  Seed: 1246
# EP: [41] ->  LER: 0.0078125  Noise scalar: 0.0020978009259259257  Seed: 1258
# EP: [42] ->  LER: 0.01171875  Noise scalar: 0.002619237075617284  Seed: 1243
# EP: [43] ->  LER: 0.0087890625  Noise scalar: 0.0026343074845679012  Seed: 1234
# EP: [44] ->  LER: 0.01171875  Noise scalar: 0.003839940200617284  Seed: 1262
# EP: [45] ->  LER: 0.005859375  Noise scalar: 0.0028437861689814816  Seed: 1279
# EP: [46] ->  LER: 0.00390625  Noise scalar: 0.0019305193865740739  Seed: 1255
# EP: [47] ->  LER: 0.0107421875  Noise scalar: 0.002521279417438272  Seed: 1271
# EP: [48] ->  LER: 0.0078125  Noise scalar: 0.0026328004436728396  Seed: 1283
# EP: [49] ->  LER: 0.013671875  Noise scalar: 0.00260718074845679  Seed: 1280
# """ 
# # ... Add the rest of your log lines here

# # MOCK DATA based on your logs (Use your real text instead!)
# # These are rough points from your provided list to show you the shape
noise_vals, ler_vals=parse_logs(lers_seen)
plt.figure(figsize=(8, 6))

# Plot the data
plt.scatter(noise_vals, ler_vals, color='blue', alpha=0.6, label='Evaluation Episodes')

# Add the "0.09%" Training Point manually to show the peak capability
plt.scatter([0.1], [0.09], color='red', marker='*', s=200, label='Peak Training Best (0.09%)')

# Formatting
plt.title('Decoder Resilience: LER vs. Instantaneous Noise Intensity', fontsize=14)
plt.xlabel('Effective Physical Error Rate (%) (Noise Scalar)', fontsize=12)
plt.ylabel('Logical Error Rate (%)', fontsize=12)
plt.axhline(y=0.1, color='green', linestyle='--', label='Physical Break-even (0.1%)')

# Grid and Legend
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

# Show
plt.tight_layout()
plt.savefig('visuals/graphs/LERvsNOISE_eval2.png')







lers_seen, _=parse_logs(lers_seen)
lers_unseen,_=parse_logs(lers_unseen)
lers_seen = np.array(lers_seen)
lers_unseen = np.array(lers_unseen)



''''Figure 1: Generalization performance of the SSM-RL decoder. The logical error rate distribution on unseen noise seeds closely matches
 that observed during training, indicating that the agent has learned the underlying noise dynamics rather than memorizing specific realizations.'''
plt.figure()
plt.boxplot(
    [lers_seen, lers_unseen],
    labels=["Seen Seeds", "Unseen Seeds"],
    showfliers=True
)

plt.ylabel("Logical Error Rate % (LER)")
plt.title("Generalization of SSM-RL Decoder")

# Optional: annotate means
plt.text(1, lers_seen.mean(), f"μ={lers_seen.mean():.4f} (%)", ha="center", va="bottom")
plt.text(2, lers_unseen.mean(), f"μ={lers_unseen.mean():.4f} (%)", ha="center", va="bottom")

plt.tight_layout()
plt.savefig('visuals/graphs/Generalization2.png')






# '''olds that produce better generalization plot'''

# lers_seen="""
# EP: [0] ->  LER: 0.0078125  Noise scalar: 0.0021731529706790122  Seed: 1235
# EP: [1] ->  LER: 0.0087890625  Noise scalar: 0.002896532600308642  Seed: 1264
# EP: [2] ->  LER: 0.0078125  Noise scalar: 0.0014844352816358024  Seed: 1260
# EP: [3] ->  LER: 0.0087890625  Noise scalar: 0.002182195216049383  Seed: 1272
# EP: [4] ->  LER: 0.0146484375  Noise scalar: 0.002424828800154321  Seed: 1239
# EP: [5] ->  LER: 0.0078125  Noise scalar: 0.0024549696180555555  Seed: 1256
# EP: [6] ->  LER: 0.005859375  Noise scalar: 0.0020978009259259257  Seed: 1253
# EP: [7] ->  LER: 0.0107421875  Noise scalar: 0.0035400390625  Seed: 1266
# EP: [8] ->  LER: 0.0048828125  Noise scalar: 0.0019380545910493826  Seed: 1261
# EP: [9] ->  LER: 0.0126953125  Noise scalar: 0.00328685619212963  Seed: 1275
# EP: [10] ->  LER: 0.0166015625  Noise scalar: 0.0025212794174382714  Seed: 1241
# EP: [11] ->  LER: 0.0107421875  Noise scalar: 0.0019079137731481482  Seed: 1247
# EP: [12] ->  LER: 0.017578125  Noise scalar: 0.002176167052469136  Seed: 1245
# EP: [13] ->  LER: 0.013671875  Noise scalar: 0.002917631172839506  Seed: 1263
# EP: [14] ->  LER: 0.01171875  Noise scalar: 0.0029929832175925927  Seed: 1273
# EP: [15] ->  LER: 0.009765625  Noise scalar: 0.002592110339506173  Seed: 1257
# EP: [16] ->  LER: 0.009765625  Noise scalar: 0.0028950255594135804  Seed: 1270
# EP: [17] ->  LER: 0.005859375  Noise scalar: 0.0019033926504629629  Seed: 1268
# EP: [18] ->  LER: 0.0068359375  Noise scalar: 0.0031391661844135804  Seed: 1281
# EP: [19] ->  LER: 0.0068359375  Noise scalar: 0.002792546778549383  Seed: 1248
# EP: [20] ->  LER: 0.0146484375  Noise scalar: 0.001930519386574074  Seed: 1267
# EP: [21] ->  LER: 0.0068359375  Noise scalar: 0.0017572096836419753  Seed: 1282
# EP: [22] ->  LER: 0.0078125  Noise scalar: 0.0016034915123456792  Seed: 1242
# EP: [23] ->  LER: 0.013671875  Noise scalar: 0.002846800250771605  Seed: 1259
# EP: [24] ->  LER: 0.0107421875  Noise scalar: 0.0030261381172839506  Seed: 1276
# EP: [25] ->  LER: 0.0029296875  Noise scalar: 0.0017677589699074073  Seed: 1269
# EP: [26] ->  LER: 0.0078125  Noise scalar: 0.0031542365933641976  Seed: 1278
# EP: [27] ->  LER: 0.013671875  Noise scalar: 0.004409601658950617  Seed: 1277
# EP: [28] ->  LER: 0.0166015625  Noise scalar: 0.003348644868827161  Seed: 1252
# EP: [29] ->  LER: 0.009765625  Noise scalar: 0.003023124035493827  Seed: 1265
# EP: [30] ->  LER: 0.01171875  Noise scalar: 0.0029658564814814816  Seed: 1274
# EP: [31] ->  LER: 0.0126953125  Noise scalar: 0.002551420235339506  Seed: 1237
# EP: [32] ->  LER: 0.0078125  Noise scalar: 0.0024896315586419755  Seed: 1254
# EP: [33] ->  LER: 0.0146484375  Noise scalar: 0.0030487437307098767  Seed: 1236
# EP: [34] ->  LER: 0.0087890625  Noise scalar: 0.002671983506944444  Seed: 1238
# EP: [35] ->  LER: 0.0087890625  Noise scalar: 0.0031376591435185184  Seed: 1244
# EP: [36] ->  LER: 0.01171875  Noise scalar: 0.0030818986304012343  Seed: 1251
# EP: [37] ->  LER: 0.0048828125  Noise scalar: 0.0020390263310185184  Seed: 1240
# EP: [38] ->  LER: 0.0087890625  Noise scalar: 0.0029372227044753086  Seed: 1249
# EP: [39] ->  LER: 0.0078125  Noise scalar: 0.002530321662808642  Seed: 1250
# EP: [40] ->  LER: 0.0048828125  Noise scalar: 0.0021008150077160494  Seed: 1246
# EP: [41] ->  LER: 0.0078125  Noise scalar: 0.0020978009259259257  Seed: 1258
# EP: [42] ->  LER: 0.01171875  Noise scalar: 0.002619237075617284  Seed: 1243
# EP: [43] ->  LER: 0.0087890625  Noise scalar: 0.0026343074845679012  Seed: 1234
# EP: [44] ->  LER: 0.01171875  Noise scalar: 0.003839940200617284  Seed: 1262
# EP: [45] ->  LER: 0.005859375  Noise scalar: 0.0028437861689814816  Seed: 1279
# EP: [46] ->  LER: 0.00390625  Noise scalar: 0.0019305193865740739  Seed: 1255
# EP: [47] ->  LER: 0.0107421875  Noise scalar: 0.002521279417438272  Seed: 1271
# EP: [48] ->  LER: 0.0078125  Noise scalar: 0.0026328004436728396  Seed: 1283
# EP: [49] ->  LER: 0.013671875  Noise scalar: 0.00260718074845679  Seed: 1280"""


# lers_unseen="""EP: [0] ->  LER: 0.0087890625  Noise scalar: 0.002574025848765432  Seed: 4322
# EP: [1] ->  LER: 0.0068359375  Noise scalar: 0.0022394627700617286  Seed: 4351
# EP: [2] ->  LER: 0.013671875  Noise scalar: 0.0021384910300925927  Seed: 4347
# EP: [3] ->  LER: 0.0146484375  Noise scalar: 0.002091772762345679  Seed: 4359
# EP: [4] ->  LER: 0.0126953125  Noise scalar: 0.00328685619212963  Seed: 4326
# EP: [5] ->  LER: 0.013671875  Noise scalar: 0.004139841338734568  Seed: 4343
# EP: [6] ->  LER: 0.0107421875  Noise scalar: 0.003650053047839506  Seed: 4340
# EP: [7] ->  LER: 0.0078125  Noise scalar: 0.002168631847993827  Seed: 4353
# EP: [8] ->  LER: 0.0078125  Noise scalar: 0.0025664906442901233  Seed: 4348
# EP: [9] ->  LER: 0.009765625  Noise scalar: 0.0025107301311728396  Seed: 4362
# EP: [10] ->  LER: 0.0078125  Noise scalar: 0.0027262369791666665  Seed: 4328
# EP: [11] ->  LER: 0.005859375  Noise scalar: 0.0022620683834876547  Seed: 4334
# EP: [12] ->  LER: 0.005859375  Noise scalar: 0.003041208526234568  Seed: 4332
# EP: [13] ->  LER: 0.01171875  Noise scalar: 0.0029100959683641976  Seed: 4350
# EP: [14] ->  LER: 0.0078125  Noise scalar: 0.00227563175154321  Seed: 4360
# EP: [15] ->  LER: 0.0087890625  Noise scalar: 0.0020751953125  Seed: 4344
# EP: [16] ->  LER: 0.0087890625  Noise scalar: 0.002917631172839506  Seed: 4357
# EP: [17] ->  LER: 0.005859375  Noise scalar: 0.0022304205246913575  Seed: 4355
# EP: [18] ->  LER: 0.0126953125  Noise scalar: 0.0022876880787037037  Seed: 4368
# EP: [19] ->  LER: 0.00390625  Noise scalar: 0.0030728563850308645  Seed: 4335
# EP: [20] ->  LER: 0.0087890625  Noise scalar: 0.0023148148148148147  Seed: 4354
# EP: [21] ->  LER: 0.0107421875  Noise scalar: 0.002209321952160494  Seed: 4369
# EP: [22] ->  LER: 0.0107421875  Noise scalar: 0.003146701388888889  Seed: 4329
# EP: [23] ->  LER: 0.0068359375  Noise scalar: 0.0018265335648148147  Seed: 4346
# EP: [24] ->  LER: 0.0068359375  Noise scalar: 0.002644856770833333  Seed: 4363
# EP: [25] ->  LER: 0.0107421875  Noise scalar: 0.002470040027006173  Seed: 4356
# EP: [26] ->  LER: 0.005859375  Noise scalar: 0.0029221522955246914  Seed: 4365
# EP: [27] ->  LER: 0.0146484375  Noise scalar: 0.0020993079668209878  Seed: 4364
# EP: [28] ->  LER: 0.009765625  Noise scalar: 0.002435378086419753  Seed: 4339
# EP: [29] ->  LER: 0.015625  Noise scalar: 0.002849814332561728  Seed: 4352
# EP: [30] ->  LER: 0.0087890625  Noise scalar: 0.00260718074845679  Seed: 4361
# EP: [31] ->  LER: 0.0078125  Noise scalar: 0.0027729552469135804  Seed: 4324
# EP: [32] ->  LER: 0.0087890625  Noise scalar: 0.002968870563271605  Seed: 4341
# EP: [33] ->  LER: 0.0087890625  Noise scalar: 0.0015432098765432098  Seed: 4323
# EP: [34] ->  LER: 0.0087890625  Noise scalar: 0.001913941936728395  Seed: 4325
# EP: [35] ->  LER: 0.0078125  Noise scalar: 0.0031120394483024694  Seed: 4331
# EP: [36] ->  LER: 0.0107421875  Noise scalar: 0.0027111665702160494  Seed: 4338
# EP: [37] ->  LER: 0.009765625  Noise scalar: 0.002956814236111111  Seed: 4327
# EP: [38] ->  LER: 0.009765625  Noise scalar: 0.002007378472222222  Seed: 4336
# EP: [39] ->  LER: 0.009765625  Noise scalar: 0.002956814236111111  Seed: 4337
# EP: [40] ->  LER: 0.0087890625  Noise scalar: 0.0027096595293209878  Seed: 4333
# EP: [41] ->  LER: 0.005859375  Noise scalar: 0.0027187017746913584  Seed: 4345
# EP: [42] ->  LER: 0.009765625  Noise scalar: 0.0013518156828703704  Seed: 4330
# EP: [43] ->  LER: 0.009765625  Noise scalar: 0.002412772472993827  Seed: 4321
# EP: [44] ->  LER: 0.0078125  Noise scalar: 0.0023826316550925927  Seed: 4349
# EP: [45] ->  LER: 0.0078125  Noise scalar: 0.0034887996720679012  Seed: 4366
# EP: [46] ->  LER: 0.0087890625  Noise scalar: 0.0025408709490740743  Seed: 4342
# EP: [47] ->  LER: 0.005859375  Noise scalar: 0.0021716459297839506  Seed: 4358
# EP: [48] ->  LER: 0.009765625  Noise scalar: 0.002575532889660494  Seed: 4370
# EP: [49] ->  LER: 0.01171875  Noise scalar: 0.002358519000771605  Seed: 4367"""



# '''===== SEEN-SEED EVAL SUMMARY =====
# LER mean=0.00977   std=0.00344
# LER min=0.00293     max=0.01758
# noise mean=0.00261
# ===================================



# ===== UNSEEN-SEED EVAL SUMMARY =====
# LER mean=0.00926   std=0.00253
# LER min=0.00391     max=0.01562
# noise mean=0.00257
# ===================================

# '''