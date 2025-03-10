import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.inverse_sparse import gamma_map, make_stc_from_dipoles
from mne.inverse_sparse.mxne_inverse import _make_sparse_stc
from mne.simulation import simulate_sparse_stc, simulate_evoked
from mne.viz import plot_sparse_source_estimates, plot_dipole_locations
from mne import convert_forward_solution
from mne.datasets import fetch_fsaverage, sample
from scipy.spatial.distance import cdist
from ot import emd2
from mne.viz import (
    plot_dipole_amplitudes,
    plot_dipole_locations,
    plot_sparse_source_estimates,
)

class DipoleSimulation:
    def __init__(self, data_path:str, subject:str, subjects_dir:str):
        self.data_path = data_path
        self.subject = subject
        self.subjects_dir = subjects_dir
        
    def create_mne_setup_source_space(self, spacing:str="oct6"):
        """ico4: 1026 Sources per hemisphere, 9.9 Source spacing / mm"""
        self.src = mne.setup_source_space(self.subject, spacing=spacing, subjects_dir=self.subjects_dir)
        
    def write_mne_source_space(self, fname:str="/data/my_ico4-src.fif", overwrite:bool=True):
        """fname: str, path to save the source space. Must end with -src.fif"""
        mne.write_source_spaces(fname, self.src, overwrite=overwrite)
    
    def read_mne_source_space(self, fname:str="/data/my_ico4-src.fif"):
        """fname: str, path to read the source space. Must end with -src.fif"""
        self.src = mne.read_source_spaces(fname)
        
    def print_src_info(self):
        print('Number of sources:', self.src[0]['nuse'], self.src[1]['nuse'])
        print('Location of the first source point (left hemisphere):', self.src[0]['rr'][0])
        print('Orientation of the first source point (left hemisphere):', self.src[0]['nn'][0])
        print("src datatype", type(self.src))
    
    def make_bem_model_and_solution(self, ico:int=4):
        """ico: int, ico grade of the BEM model"""
        self.bem_model = mne.make_bem_model(subject=self.subject, ico=4, subjects_dir=self.subjects_dir)
        self.bem = mne.make_bem_solution(self.bem_model)
        
        
    def make_forward_solution(self, info, src, bem, trans:str='fsaverage', eeg:bool=True, meg:bool=False):
        self.fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, eeg=True, meg=False)
    
    def extract_leadfield(self, fwd):
        self.leadfield = fwd["sol"]["data"]
    
    def print_fwd_info(self, fwd, leadfield):
        print("Leadfield size : %d (channels) x %d (3*nsource)" % leadfield.shape)
        assert leadfield.shape[1]//3 == fwd['nsource']
        
        fwd_src = fwd['src']
        print('Number of sources:', fwd_src[0]['nuse'], fwd_src[1]['nuse'])
        print('Location of the first source point (left hemisphere):', fwd_src[0]['rr'][0])
        print('Orientation of the first source point (left hemisphere):', fwd_src[0]['nn'][0])
        
        print("fwd_src type:", type(fwd_src))
        
        vertices = [s['vertno'] for s in self.src] 
        n_sources = sum(len(v) for v in vertices)
        n_sources, vertices
        print(f'fwd: nr sources in both him: {fwd["nsource"], fwd["nchan"]}')
        print(f'fwd: sum of nr of sources in each him: {fwd_src[0]['nuse'] + fwd_src[1]['nuse']}')

if __name__ == "__main__":
    data_path = sample.data_path()
    subject = "fsaverage"
    subjects_dir = data_path / "subjects"
    
    ds = DipoleSimulation(data_path=data_path, subject=subject, subjects_dir=subjects_dir)
    ds.create_mne_setup_source_space()
    ds.write_mne_source_space()
    ds.read_mne_source_space()
    ds.print_src_info()
    ds.make_bem_model_and_solution()
    ds.make_forward_solution()
    ds.extract_leadfield()
    ds.print_fwd_info()
    
    