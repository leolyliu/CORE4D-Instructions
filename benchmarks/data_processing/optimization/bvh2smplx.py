import os
from os.path import join, dirname, abspath
import sys
sys.path.insert(0, join(dirname(abspath(__file__)), "../.."))
from data_processing.smplx import smplx


def create_SMPLX_model(smplx_model_dir="/share/human_model/models", use_pca=True, num_pca_comps=12, batch_size=1, device="cuda:0"):
    """
    create an SMPLX model
    num_pca_comps: hand PCA dimension
    """
    smplx_model = smplx.create(smplx_model_dir, model_type="smplx", gender="neutral", batch_size=batch_size, use_face_contour=False, num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=use_pca, num_pca_comps=num_pca_comps, flat_hand_mean=True)
    smplx_model.to(device)
    return smplx_model
