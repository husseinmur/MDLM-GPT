# angle_utils.py
import numpy as np
from string import ascii_lowercase
from numpy.random import uniform

class AngleProcessor:
    @staticmethod
    def get_decode_ranges():
        bins = [-3.140653, -2.72200405, -2.30335511, -1.88470616, -1.46605721,
                -1.04740827, -0.62875932, -0.21011037, 0.20853857, 0.62718752,
                1.04583647, 1.46448541, 1.88313436, 2.30178331, 2.72043225,
                3.1390812]
        labels = ascii_lowercase[:15]
        return {labels[i]: (bins[i], bins[i+1]) for i in range(len(bins)-1)}

    @staticmethod
    def decode_conf_with_noise(conf, max_noise=0):
        phis, psis = [], []
        ranges = AngleProcessor.get_decode_ranges()
        
        for angle in conf:
            phi = uniform(*ranges[angle[1]])
            psi = uniform(*ranges[angle[2]])
            if max_noise > 0:
                phi += np.random.normal(0.1, max_noise)
                psi += np.random.normal(0.1, max_noise)
                phi = np.clip(phi, -np.pi, np.pi)
                psi = np.clip(psi, -np.pi, np.pi)
            phis.append(phi)
            psis.append(psi)
        return phis, psis

    @staticmethod
    def calculate_conformation_loss(phis, psis, kdes, fasta):
        loss = 0
        fasta_pairs = [fasta[i]+fasta[i+1] for i in range(len(fasta)-1)]
        for pair, phi, psi in zip(fasta_pairs, phis, psis):
            if pair in kdes.keys():
                prob = kdes[pair].evaluate_points(phi, psi)
                loss += -np.log(prob)
        return loss/10