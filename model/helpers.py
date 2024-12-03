from string import ascii_lowercase
import numpy as np
from scipy.interpolate import interp1d
import math


def smart_restore(tokens, bins):
    if len(bins) == 2:  # Only two bins, meaning all angles are the same
        return [np.full_like(frame_tokens, bins[0], dtype=float) for frame_tokens in tokens]
    
    # Create mid-points of bins
    mid_points = (bins[:-1] + bins[1:]) / 2
    
    # Add circular continuity
    extended_mid_points = np.concatenate([[mid_points[-1] - 360], mid_points, [mid_points[0] + 360]])
    extended_tokens = np.arange(len(extended_mid_points))
    
    # Create interpolation function
    interp_func = interp1d(extended_tokens, extended_mid_points, kind='linear')
    
    # Restore angles with some random jitter
    restored_angles = []
    for frame_tokens in tokens:
        jitter = np.random.uniform(-0.4, 0.4, size=len(frame_tokens))
        frame_restored = interp_func(frame_tokens + 1 + jitter)  # +1 because we added one point at the beginning
        frame_restored = np.mod(frame_restored + 180, 360) - 180
        restored_angles.append(frame_restored)
    
    return restored_angles

def get_bins():
    phi_bins = np.array([-180.        , -151.70072095, -134.51679433, -118.53060149,
       -105.24655944,  -94.62668651,  -85.53657852,  -76.91966846,
        -67.69657065,  -56.67403355,  -41.97324966,   12.22099106,
         67.35450638,   84.41166979,  131.07917311,  180.        ])
    psi_bins = np.array([-180.        , -140.36917804,  -88.81258903,  -54.58659278,
        -36.6442538 ,  -21.35289848,   -4.30648714,   16.32787949,
         39.56355957,   77.53086595,  111.06948336,  125.10818678,
        137.94519818,  151.00803   ,  164.09803119,  180.        ])
    all_letters = ascii_lowercase

    return phi_bins, psi_bins, all_letters

def decode_conf(conf):
    phi_bins, psi_bins, all_letters = get_bins()
    phis=[]
    psis=[]
    for pair in conf:
        phi = all_letters.index(pair[1])
        phis.append(math.radians(smart_restore([np.array([phi])],phi_bins)[0][0]))
        psi = all_letters.index(pair[2])
        psis.append(math.radians(smart_restore([np.array([psi])],psi_bins)[0][0]))
    return phis,psis

def calculate_conformation_loss(conf, kdes, fasta):
    # low memory version of the loss function
    loss = 0
    fasta_pairs = [fasta[i]+fasta[i+1] for i in range(len(fasta)-1)]
    phis, psis = decode_conf(conf)
    for pair, phi, psi in zip(fasta_pairs, phis, psis):
        if pair in kdes.keys():
            prob = kdes[pair]([phi, psi])[0]
            if prob < 0.01 and phi<2:
                loss +=0.001
        else:
            continue
    return loss


def get_angles_loss(logits, tokenizer, kdes, fasta):
    nbatches = logits.shape[0]
    seq_length = logits.shape[1]
    vocab_size = logits.shape[2]
    logits = logits.reshape(nbatches * seq_length, vocab_size).cpu().data.numpy()
    confs = np.argmax(logits, axis=1).reshape(-1,9)
    total_loss = 0
    for conf in confs:
        conf = tokenizer.decode(conf, skip_special_tokens=True).split()
        if len(conf)==9:
            total_loss += calculate_conformation_loss(conf, kdes, fasta)
        else:
            total_loss += 1
    return total_loss

