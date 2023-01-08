###
# MODULO PER I COLORI
###

import numpy as np

from matplotlib.colors import ListedColormap

import os.path


def load_cmap(name, dirn = None):
    if dirn is None:
        dirn = os.path.join( os.path.dirname(__file__), "cmaps" )
    
    with open(os.path.join(dirn, name + ".cmap"), 'r') as cmapfile:
        hexstr = cmapfile.read().replace('\n', '')

    return build_cmap(hexstr, name)

def save_cmap(cmap, fname = None, overwrite = False):
    if fname is None:
        dirn = os.path.join( os.path.dirname(__file__), "cmaps" )
        fname = os.path.join(dirn, cmap.name + ".cmap")

    try:
        if overwrite:
            f = open(fname, 'w')
        else:
            f = open(fname, 'x')

        raise NotImplementedError()
    except FileExistsError:
        pass

def build_cmap(hexstr, name = None):
    hexv  = [hs for hs in hexstr.replace(',', '').split('#') if len(hs) > 0]
    rgbs = np.zeros((len(hexv), len(hexv[0])//2))
    for i,cstr in enumerate(hexv):
        rgbs[i] = np.array([ int(cstr[i:i+2],16) for i in range(0,len(hexv[0]),2) ])
    rgbs = rgbs/255 # scale in [0,1]
    
    if name is None: name = hexv[0] + "-" + hexv[-1]
    
    return ListedColormap(rgbs, name = name)