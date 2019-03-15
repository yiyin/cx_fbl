#!/usr/bin/env python

"""
Assign neuron and synapse model parameters.
"""

import logging
import sys

import numpy as np
from pyorient.ogm import Graph, Config

import neuroarch.models as models
import neuroarch.query as query
import neuroarch.nxtools as nxtools

from cx_config import cx_db, model_version
graph = Graph(Config.from_url(cx_db, 'root', 'root',
                              initial_drop=False))
graph.include(models.Node.registry)
graph.include(models.Relationship.registry)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('cx')

def leaky_iaf_params(lpu, extern):
    """
    Generate LeakyIAF params.
    """
    k = 1000
    assert isinstance(extern, bool)
    if lpu == 'BU' or lpu == 'bu':
        return {'extern': extern,
                'initV': -0.06 * k,
                'reset_potential': -0.0675489770451* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.001* k,
                'resistance': 1.02445570216* k,
                'capacitance': 0.0669810502993,
                'resting_potential': 0.0* k}
    elif lpu == 'EB':
        return {'extern': extern,
                'initV': -0.06* k,
                'reset_potential': -0.0675489770451* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.001* k,
                'resistance': 1.02445570216* k,
                'capacitance': 0.0669810502993,
                'resting_potential': 0.0* k}
    elif lpu == 'FB':
        return {'extern': extern,
                'initV': -0.06* k,
                'reset_potential': -0.0675489770451* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.001* k,
                'resistance': 1.02445570216* k,
                'capacitance': 0.0669810502993,
                'resting_potential': 0.0* k}
    elif lpu == 'PB':
        return {'extern': extern,
                'initV': -0.06* k,
                'reset_potential': -0.07* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.00001* k,
                'resistance': 0.25* k,
                'capacitance': 0.5,
                'resting_potential': 0.0* k}
    elif lpu in ['LAL', 'lal']:
        return {'extern': extern,
                'initV': -0.06* k,
                'reset_potential': -0.07* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.00001* k,
                'resistance': 0.25* k,
                'capacitance': 0.5,
                'resting_potential': 0.0* k}
    elif lpu in ['NO', 'no']:
        return {'extern': extern,
                'initV': -0.06* k,
                'reset_potential': -0.07* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.00001* k,
                'resistance': 0.25* k,
                'capacitance': 0.5,
                'resting_potential': 0.0* k}
    else:
        raise ValueError('unrecognized LPU name')

def alpha_synapse_params(lpu):
    """
    Generate AlphaSynapse params.
    """
    k = 1000
    s = 0.001
    if lpu == 'BU' or lpu == 'bu':
        return {'conductance': True,
                'ad': 0.16*1000,
                'ar': 1.1*100,
                'gmax': 0.01 * s,
                'reverse': -0.065 * k}
    elif lpu == 'EB':
        return {'conductance': True,
                'ad': 0.16*1000,
                'ar': 1.1*100,
                'gmax': 0.01 * s * 0.3,
                'reverse': 0.065* k}
    elif lpu == 'FB':
        return {'conductance': True,
                'ad': 0.16*1000,
                'ar': 1.1*100,
                'gmax': 0.01 * s,
                'reverse': 0.065* k}
    elif lpu == 'PB':
        return {'conductance': True,
                'ad': 0.19*1000,
                'ar': 1.1*100,
                'gmax': 0.002 * s,
                'reverse': 0.065* k}
    elif lpu in ['LAL', 'lal']:
        return {'conductance': True,
                'ad': 0.19*1000,
                'ar': 1.1*100,
                'gmax': 0.002 * s,
                'reverse': 0.065* k}
    elif lpu in ['NO', 'no']:
        return {'conductance': True,
                'ad': 0.19*1000,
                'ar': 1.1*100,
                'gmax': 0.002 * s,
                'reverse': 0.065* k}
    elif lpu in ['CRE', 'cre']:
        return {'conductance': True,
                'ad': 0.19*1000,
                'ar': 1.1*100,
                'gmax': 0.002 * s,
                'reverse': 0.065* k}
    else:
        raise ValueError('unrecognized LPU name')

if __name__ == '__main__':

# Get all LeakyIAF/AlphaSynapse nodes in each LPU:
    lpu_list = [
                'PB','FB','EB','NO', 'no',
                'BU','bu', 'LAL', 'lal', 'cre', 'CRE'
                ]
    lpu_to_query = {}
    for lpu in lpu_list:
        logger.info('retrieving LeakyIAF/AlphaSynapse nodes for LPU %s' % lpu)
        lpu_node = graph.LPUs.query(name = lpu,
                                    version = model_version).one() #TODO
        lpu_to_query[lpu] = lpu_node.owns(1, cls = ['LeakyIAF', 'AlphaSynapse'])

    # Assign parameters:
    for lpu in lpu_list:
        for n in lpu_to_query[lpu].nodes_as_objs:
            if isinstance(n, models.LeakyIAF):
                logger.info('assigning params to %s LeakyIAF %s' % (lpu, n.name))
                n.update(**leaky_iaf_params(lpu, True))
            elif isinstance(n, models.AlphaSynapse):
                logger.info('assigning params to %s AlphaSynapse %s' % (lpu, n.name))
                n.update(**alpha_synapse_params(lpu))

    # Rerun queries to fetch updated data:
    for lpu in lpu_list:
        logger.info('rerunning query for LPU %s to fetch updated data' % lpu)
        lpu_to_query[lpu].execute(True, True)
