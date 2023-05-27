##@Author: Gemma Gordon
##@Date: July 2022
## Functions used to determine the epitope concavity of antigen structures bound by Abs and sdAbs


import ast
from Bio import PDB
from Bio.PDB.PDBList import PDBList
import os
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
import numpy as np
import pandas as pd
import open3d as o3d
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import shutil
from Bio.SeqUtils import seq3

parser = PDB.PDBParser()
io = PDB.PDBIO()
pdbl = PDBList()

# load in data with epitope residue data 
nbs_dataset = pd.read_csv('gg_sdabs_summary_epitopes.csv')
abs_dataset = pd.read_csv('gg_flabs_summary_epitopes.csv')

# for converting .ent to .pdb files
def ent_to_pdb(base_file):

    name, ext = base_file.rsplit('.', 1)
    new_file = '{}.{}'.format(name, 'pdb')

    with open(base_file , 'r') as f1:
        with open(new_file, 'w') as f2:
            f2.write(f1.read())
            os.remove(base_file)

    return new_file


def get_pdb(pdb_file): # NOTE changed from pdb_id to pdb_file

    # retrieve strucure based on ID
    # pdb_file = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb', overwrite=True, pdir='temp_pdb') # NOTE removed for Ag chains, already have files
    
    # get pdb ID from file name
    pdb_id = str(pdb_file).split('b')
    pdb_id = pdb_id[1].split('_')
    pdb_id = pdb_id[0]
    
    #pdb_file = ent_to_pdb(pdb_file)
    struct = parser.get_structure(pdb_id, pdb_file)

    return struct, pdb_id

def get_all_coords(struct):

    X = []
    Y = []
    Z = []

    counter = 0
    for model in struct:
        for chain in model:
                for residue in chain:
                            for atom in residue:
                                counter += 1
                                _x, _y, _z = atom.get_coord()
                                X.append(_x)
                                Y.append(_y)
                                Z.append(_z)

    all_coords = np.array([X,Y,Z]).T

    return all_coords


def get_epitope_residues(pdb_id, dataset_df, column_name):

    # get epitope residues by processing summary data from CV experiment
    epitope_residues = dataset_df[column_name].loc[dataset_df['pdb']== pdb_id].iloc[0] # NOTE change to epitope_arp_VHVL if Abs dataset
    epitope_residues = ast.literal_eval(epitope_residues)

    # get residue position numbers
    resnb = list(epitope_residues.keys())
    # get residue types and convert from 1 letter to 3 letter format
    restype1 = list(epitope_residues.values())
    restype = []
    for item in restype1:
        item = seq3(item).upper()
        restype.append(item)

    return resnb, restype, epitope_residues


def get_epitope_coords(resnbs, restypes, struct, epitope_residues):

    x,y,z = [],[],[]

    epitope_res = []
    for model in struct:
        for chain in model:
            for residue in chain:
                for resnb, restype in zip(resnbs, restypes):
                    if str('resseq=' + resnb) in str(residue) and restype in str(residue):
                        if len(epitope_res) < len(epitope_residues):
                            epitope_res.append(residue)

    for residue in epitope_res:
        for atom in residue:
            _x, _y, _z = atom.get_coord()
            x = np.append(x, _x)
            y = np.append(y, _y)
            z = np.append(z, _z)

    epitope_points = np.array([x,y,z]).T

    return epitope_points


def create_structure_mesh(all_coords):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_coords)

    # NOTE quick method to find a better alpha value, won't find exact best value but closer than before
    mesh_dict = dict()
    for alpha in np.arange(0,3.1,0.1): # NOTE was set at (0,10,0.5) and all structures gave alpha values from 1-3 (mostly 2.0, 2.5)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        triangles = len(mesh.triangles)
        mesh_dict[alpha] = triangles

    # get alpha value where triangles highest number
    max_triangles = max(mesh_dict.values())
    for key in mesh_dict.keys():
        if mesh_dict[key] == max_triangles:
            print(key)
            alpha = key

    print('ALPHA = ', alpha)
    #mesh.compute_vertex_normals() # NOTE - need this?
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh_vertices = np.asarray(mesh.vertices)
    # save mesh in temp file to reload as trimesh mesh object (as opposed to open3d mesh)
    o3d.io.write_triangle_mesh('temp_mesh.ply', mesh, print_progress=True)

    return mesh_vertices, alpha


def load_trimesh_mesh(file_name):

    mesh = trimesh.load(file_name)

    return mesh


def map_mesh_point_clouds(all_coords, mesh_vertices):

    '''map points in original PDB epitope structure to those from mesh vertices'''

    # make kdtree
    kdtree = KDTree(all_coords)
    # find closest points of mesh coords to those from PDB coords
    mapped_points = kdtree.query(mesh_vertices, k=1)
    mapped_index = mapped_points[1] # gives indices of all_coords points that the mesh points map to closest
    mapped_mesh_points = all_coords[mapped_index]

    return mapped_mesh_points


def get_vertex_defects(mesh, mapped_mesh_points):

    '''gives mesh points of structure alongside corresponding vertex defects values'''

    vd_values = trimesh.curvature.vertex_defects(mesh)
    vd_df = pd.DataFrame(mapped_mesh_points, columns=['x','y','z'])
    vd_df['vertex defects'] = pd.Series(vd_values)

    return vd_df


def map_epitope_to_mesh(mapped_mesh_points, epitope_points):

    # make kdtree of the mesh points that were mapped to the PDB coords
    # NOTE confused here - is double mapping necessary?
    kdtree = KDTree(mapped_mesh_points)
    # map epitope PDB coords to the mesh
    epitope_map = kdtree.query(epitope_points, k=1)
    df_query_index = epitope_map[1] # these indexes used to extract VD from results dataframe

    return df_query_index, epitope_map


## FROM ROTATION 1 
def rmsd(V, W): # from https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py 
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.
    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    diff = np.array(V) - np.array(W)
    N = len(V)
    return np.sqrt((diff * diff).sum() / N)


def align_coords(query_points, ref_points):
    qm_means = query_points.mean(0)
    rm_means = ref_points.mean(0)
    qm_trans = query_points - qm_means
    rm_trans = ref_points - rm_means

    trans_vector = rm_means # add to revert_translation

    rot_matrix, __ = Rotation.align_vectors(a=rm_trans, b=qm_trans)
    rot_matrix = rot_matrix.as_matrix() # scipy gives Rotation object, get matrix
    qm_rotated = qm_trans  @ rot_matrix

    # qm_reverted, frm_reverted = revert_translation(qm_rotated, frm_trans, qm_means, frm_means)
    qm_aligned =  qm_rotated + trans_vector
    rmsd_val = rmsd(qm_aligned, rm_trans + trans_vector)
    return qm_aligned, rmsd_val


## back to rotation 2
def check_rmsd(epitope_points, mapped_mesh_points, epitope_map):

    '''check how well epitope from PDB mapped to mesh vertices points'''

    pc1 = epitope_points
    pc2 = mapped_mesh_points[epitope_map[1]]
    pc2_aligned, rmsd_val = align_coords(pc1, pc2)

    return rmsd_val


def get_epitope_vd_values(vd_df, df_query_index):

    epitope_vd_values = vd_df['vertex defects'][df_query_index].values
    mean_vd = np.mean(epitope_vd_values)

    positive = 0
    negative = 0

    for value in epitope_vd_values:
        if value > 0:
            positive += 1
        elif value < 0:
            negative += 1

    # proportion of positive and negative values
    positive_prop = positive / (positive + negative)
    negative_prop = negative / (positive + negative)
    proportions = dict()
    proportions['positive VD %'] = positive_prop
    proportions['negative VD %'] = negative_prop

    return epitope_vd_values, mean_vd, proportions



# CALL FUNCTIONS for all pdb structures in Nbs or Abs dataset, add results to df
dataset = nbs_dataset # NOTE change when running Nbs or Abs

# create df to hold results
results_df = pd.DataFrame()

pdb_ids = []
mean_vd_values = []
all_epitope_vd_values = []
all_vd_proportions = []
rmsd_values = []
alpha_values = []

failed = [] 
fail_count = 0
for pdb_file in os.listdir('sdabs_ag_chains'):

    try:
        print('Running on', pdb_file)

        # open3d
        struct, pdb_id = get_pdb(pdb_file) # NOTE changed from pdb ID to pdb file
        all_coords = get_all_coords(struct)
        resnbs, restypes, epitope_residues = get_epitope_residues(pdb_id, dataset_df=nbs_dataset, column_name='epitope_arp') # NOTE change col name
        epitope_points = get_epitope_coords(resnbs, restypes, struct, epitope_residues)
        mesh_vertices, alpha = create_structure_mesh(all_coords)

        # trimesh
        mesh = load_trimesh_mesh(file_name='temp_mesh.ply')
        mapped_mesh_points = map_mesh_point_clouds(all_coords, mesh_vertices)
        vd_df = get_vertex_defects(mesh, mapped_mesh_points)
        df_query_index, epitope_map = map_epitope_to_mesh(mapped_mesh_points, epitope_points)
        rmsd_val = check_rmsd(epitope_points, mapped_mesh_points, epitope_map)
        epitope_vd_values, mean_vd, proportions = get_epitope_vd_values(vd_df, df_query_index)

        # add results to lists
        pdb_ids.append(pdb_id)
        mean_vd_values.append(mean_vd)
        all_epitope_vd_values.append(epitope_vd_values)
        all_vd_proportions.append(proportions)
        rmsd_values.append(rmsd_val)
        alpha_values.append(alpha)

    except:
        fail_count += 1
        failed.append(pdb_id)
        pass


# add results to df and create csv
results_df['pdb'] = pdb_ids
results_df['all_epitope_vd_values'] = all_epitope_vd_values
results_df['mean_VD'] = mean_vd_values
results_df['RMSD (PDB vs mesh)'] = rmsd_values
results_df['positive vs negative'] = all_vd_proportions
results_df['alpha'] = alpha_values

results_df.to_csv('mock.csv')

print('FAIL COUNT =', fail_count)
print(failed)


