from tqdm import tqdm
import pandas as pd
import numpy as np
import random

class ConformationSampler:
    def __init__(self, model_setup, angle_processor, batch_size=36):
        self.model_setup = model_setup
        self.angle_processor = angle_processor
        self.batch_size = batch_size

    def generate_sequences(self, input_ids):
        return self.model_setup.model.generate(
            input_ids,
            max_length=252,
            do_sample=True,
            num_beams=2,
            no_repeat_ngram_size=2,
            num_return_sequences=1,
            temperature=2.0,
            pad_token_id=self.model_setup.tokenizer.pad_token_id
        )
    
    def accept_state(self, ref_conf, new_conf, kdes, fasta, T=300):
        # Get angles and calculate energies
        ref_phis, ref_psis = self.angle_processor.decode_conf_with_noise(ref_conf)
        ref_energy = self.angle_processor.calculate_conformation_loss(ref_phis, ref_psis, kdes, fasta)
        
        new_phis, new_psis = self.angle_processor.decode_conf_with_noise(new_conf, 1.0)
        new_energy = self.angle_processor.calculate_conformation_loss(new_phis, new_psis, kdes, fasta)
        
        # Calculate energy difference and acceptance probability
        delta_energy = new_energy - ref_energy
        kT = 0.001987 * T  # Boltzmann constant in kcal/mol/K
        
        acceptance_prob = 1 / (np.exp(delta_energy/kT) + 1)
        
        return (random.random() < acceptance_prob), new_phis, new_psis


    def sample_conformations(self, train_data, num_samples, kdes, fasta, output_path='conformations.npy'):
        start = train_data[list(range(9))].to_numpy()
        num_batches = num_samples // self.batch_size
        accepted = []

        for _ in tqdm(range(num_batches)):
            idx = np.random.randint(0, len(start), self.batch_size)
            start_selected = start[idx]
            input_strs = [" ".join(row) for row in start_selected]
            
            input_ids = self.model_setup.tokenizer(
                input_strs, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).input_ids.to("cuda")
            
            output_sequences = self.generate_sequences(input_ids)
            batches = output_sequences.reshape(-1, 28, 9)
            
            self._process_batch(batches, idx, start, accepted, kdes, fasta)
        
        accepted_array = np.array(accepted)
        
        np.save(output_path, accepted_array)
        
        return accepted

    def _process_batch(self, batches, idx, start, accepted, kdes, fasta):
        for i, batch in enumerate(batches):
            ref_conf = self.model_setup.tokenizer.decode(
                batch[0], 
                skip_special_tokens=False, 
                lowercase=False
            ).split()
            
            for seq in batch[1:]:
                conf = self.model_setup.tokenizer.decode(
                    seq, 
                    skip_special_tokens=False, 
                    lowercase=False
                ).split()
                
                passed, new_phis, new_psis = self.accept_state(
                    ref_conf, conf, kdes, fasta
                )
                
                if passed:
                    start[idx[i]] = conf
                    accepted.append((new_phis, new_psis))
                    break
                else:
                    ref_phis, ref_psis = self.angle_processor.decode_conf_with_noise(ref_conf)
                    accepted.append((ref_phis, ref_psis))