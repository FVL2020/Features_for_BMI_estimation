from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import math
from absl import flags
import numpy as np
from scipy.spatial import ConvexHull
from collections import OrderedDict

import skimage.io as io
import tensorflow as tf
import cv2

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if np.max(img.shape[:2]) != config.img_size:
        print('Resizing so the max image size is %d..' % config.img_size)
        scale = (float(config.img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]
    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    crop = 2 * ((crop / 255.) - 0.5)
    return crop, proc_param, img

def save_obj(outmesh_path, vert, face):
    with open(outmesh_path, 'w') as fp:
        for v in vert:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in face + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    print('The obj model has already saved.')

def calculate_surface(vert,face):
    surface = 0
    for j in range(1,13776):
        p1 = face[j,0]
        p2 = face[j,1]
        p3 = face[j,2]
        x1 = vert[p1,0]
        y1 = vert[p1,1]
        z1 = vert[p1,2]
        x2 = vert[p2,0]
        y2 = vert[p2,1]
        z2 = vert[p2,2]
        x3 = vert[p3,0]
        y3 = vert[p3,1]
        z3 = vert[p3,2]
        a = math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
        b = math.sqrt((x1-x3)**2+(y1-y3)**2+(z1-z3)**2)
        c = math.sqrt((x2-x3)**2+(y2-y3)**2+(z2-z3)**2)
        p = (a+b+c)/2
        s = math.sqrt(p*(p-a)*(p-b)*(p-c))
        surface = surface + s
    return surface
    
def extract(filename, vert, face):
    values = {}
    x = vert[:,0]
    y = vert[:,1]
    z = vert[:,2]
    values['maxLength'] = max(x)-min(x)
    values['maxWidth'] = max(y)-min(y)
    values['maxHeigth'] = max(z)-min(z)
    
    cov_mat = np.cov(np.transpose(vert))
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    eigen_vals = sorted(eigen_vals)
    values['eigenvalue1'] =  eigen_vals[2]
    values['eigenvalue2'] =  eigen_vals[1]
    values['eigenvalue3'] =  eigen_vals[0]
    sum_eig = sum(eigen_vals)
    values['sphericity'] = eigen_vals[0]/sum_eig;
    values['flatness'] = 2*(eigen_vals[1]-eigen_vals[0])/sum_eig
    values['linearity'] = (eigen_vals[2]-eigen_vals[1])/sum_eig
    
    x0 = np.mean(x)
    y0 = np.mean(y)
    z0 = np.mean(z)
    sum_d = 0
    sum_d2 = 0
    #sum_d4 = 0;
    for i in range(1,6890):
        sum_d = sum_d + math.sqrt((x[i]-x0)**2 + (y[i]-y0)**2 + (z[i]-z0)**2)
        sum_d2 = sum_d2 + (x[i]-x0)**2 + (y[i]-y0)**2 + (z[i]-z0)**2
        #sum_d4 = sum_d4 + ((x[i]-x0)**2 + (y[i]-y0)**2 + (z[i]-z0)**2)**2
    values['compactness'] = math.sqrt(sum_d2/6890)
    values['kurtosis'] = sum_d/6890
    #alt_compactness = sum_d4/flatness
    
    values['volume'] = ConvexHull(vert).volume
    values['surface'] = calculate_surface(vert, face)
    ThreeDFeatures[filename] = values

def visualize(img, proc_param, joints, verts, cam, name, filename):
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])
    outmesh_path = 'replace_with_your_path/Obj_Models/PressMap/'+name+'.obj'
    #outmesh_path = 'replace_with_your_path/Obj_Models/RGB/train'+name+'.obj'
    #outmesh_path = 'replace_with_your_path/Obj_Models/RGB/test'+name+'.obj'
    #outmesh_path = 'replace_with_your_path/Obj_Models/RGB/val'+name+'.obj'
    vert = vert_shifted
    face = renderer.faces
    save_obj(outmesh_path,vert,face)
    extract(filename, vert, face)

path = 'replace_with_your_path/Datasets/PmatData_Supine'
#path = 'replace_with_your_path/Datasets/RGB/Image_train'
#path = 'replace_with_your_path/Datasets/RGB/Image_test'
#path = 'replace_with_your_path/Datasets/RGB/Image_val'
path_list=os.listdir(path)
path_list.sort()
config = flags.FLAGS
config(sys.argv)
config.load_path = src.config.PRETRAINED_MODEL
config.batch_size = 1
renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
ThreeDFeatures = {}

for filename in path_list:
    tf.reset_default_graph()
    sess = tf.Session()
    img_path = os.path.join(path,filename)
    name = filename[:-4]
    model = RunModel(config, sess=sess)
    input_img, proc_param, img = preprocess_image(img_path)
    input_img = np.expand_dims(input_img, 0)
    joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)
    visualize(img, proc_param, joints[0], verts[0], cams[0], name, filename)

json_path = os.path.join('replace_with_your_path/3D_Features', 'ThreeDFeatures_PressMap.json') 
#json_path = os.path.join('replace_with_your_path/3D_Features', 'ThreeDFeatures_RGB_train.json')
#json_path = os.path.join('replace_with_your_path/3D_Features', 'ThreeDFeatures_RGB_test.json')
#json_path = os.path.join('replace_with_your_path/3D_Features', 'ThreeDFeatures_RGB_val.json')
json_str = json.dumps(ThreeDFeatures)
with open(json_path, 'w') as json_file:
    json_file.write(json_str)