import csv
import json
import re
import sys
import pickle
import itertools
import logging

import path

import numpy as np
import numpy as np
from pyorient.ogm import Graph, Config

import neuroarch.models as models
import neuroarch.query as query
import neuroarch.nxtools as nxtools

from cx_config import cx_db, cx_version, model_version

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('cx')

logger.info('Start to Load Model Data')

graph = Graph(Config.from_url(cx_db, 'root', 'root', initial_drop = False))
graph.create_all(models.Node.registry)
graph.create_all(models.Relationship.registry)

CX_Neuropils = [
                'PB','FB','EB','NO', 'no',
                'BU','bu', 'LAL', 'lal', 'cre', 'CRE'
                ]

q = query.QueryWrapper(graph,
                       query.QueryString(
                         "select from Neuropil where name in {} and version='{}'".format(
                              "['"+"','".join(CX_Neuropils)+"']", cx_version),
                         'sql'))
neuropils, _ = q.get_as('obj')

lpu_dict = {}   # map LPU name to LPU nodes
lpu_port_counter = {}

for neuropil in neuropils:
    # create LPUs
    lpu = graph.LPUs.create(name = neuropil.name,
                            version = model_version)
    graph.Models.create(lpu, neuropil, version = model_version) # need to add version?
    # create LPU Interfaces
    interface = graph.Interfaces.create(name = neuropil.name)
    graph.Owns.create(lpu, interface)
    lpu_dict[lpu.name] = {'LPU': lpu, 'Interface': interface}

    lpu_port_counter[lpu.name] = {'out': itertools.count(),
                                  'in': itertools.count()}

logger.info('Created LPUs')


CX_Tracts = list(set(['-'.join(sorted(a))for a in itertools.product(CX_Neuropils, CX_Neuropils)])\
            - set(['-'.join(a) for a in zip(CX_Neuropils, CX_Neuropils)]))
q_tract = query.QueryWrapper(graph,
                       query.QueryString(
                         "select from Tract where name in {} and version='{}'".format(
                              "['"+"','".join(CX_Tracts)+"']", cx_version),
                         'sql'))
tracts, _ = q_tract.get_as('obj')
pattern_dict = {}

for tract in tracts:
    linked_neuropils = sorted(list(tract.neuropils))
    pattern_pair = frozenset(linked_neuropils)

    pattern = graph.Patterns.create(name = tract.name,
                                    version = model_version,
                                    LPUs = tract.neuropils)
    graph.Models.create(pattern, tract, version = model_version)

    int_0 = graph.Interfaces.create(
        name = '{}/{}'.format(tract.name, linked_neuropils[0]))
    int_1 = graph.Interfaces.create(
        name = '{}/{}'.format(tract.name, linked_neuropils[1]))
    graph.Owns.create(pattern, int_0)
    graph.Owns.create(pattern, int_1)
    pattern_dict[pattern_pair] = {'pattern': pattern,
                                  linked_neuropils[0]: int_0,
                                  linked_neuropils[1]: int_1}

logger.info('Created Patterns')

for neuropil in neuropils:
    neuropil_name = neuropil.name
    lpu = lpu_dict[neuropil_name]['LPU']
    q_neuropil = query.QueryWrapper.from_objs(graph, [neuropil])
    q_neurons = q_neuropil.traverse_owns(max_levels = 1, cls = 'Neuron')
    neurons, _ = q_neurons.get_as('obj')
    for neuron in neurons:
        exec_node = graph.LeakyIAFs.create(name = neuron.name)
        graph.Models.create(exec_node, neuron, version = model_version)
        graph.Owns.create(lpu, exec_node)

        q_neuron = query.QueryWrapper.from_objs(graph, [neuron])
        synapse_loc, _ = q_neuron.gen_traversal_out(
                            ['SendsTo', 'Synapse', 'instanceof'],
                            min_depth = 1).owned_by(
                                cls = 'Neuropil',
                                version = cx_version).get_as('obj')
        if any([loc.name != neuropil_name for loc in synapse_loc]):
            sel_i = '/{}/out/spk/{}'.format(
                        neuropil_name,
                        next(lpu_port_counter[neuropil_name]['out']))
            port_i = graph.Ports.create(selector = sel_i,
                                        port_io = 'out',
                                        port_type = 'spike',
                                        neuron = neuron.name)
            # Interface nodes must own the new Port nodes:
            graph.Owns.create(lpu_dict[neuropil_name]['Interface'], port_i)
            graph.SendsTo.create(exec_node, port_i)
            for loc in synapse_loc:
                if loc.name != neuropil_name:
                    sel_j = '/{}/in/spk/{}'.format(
                            loc.name, next(lpu_port_counter[loc.name]['in']))
                    port_j = graph.Ports.create(selector = sel_j,
                                                port_io = 'in',
                                                port_type = 'spike',
                                                neuron = neuron.name)
                    graph.Owns.create(lpu_dict[loc.name]['Interface'], port_j)

                    pattern_pair = frozenset([neuropil_name, loc.name])
                    pattern = pattern_dict[pattern_pair]['pattern']
                    int_0 = pattern_dict[pattern_pair][neuropil_name]
                    int_1 = pattern_dict[pattern_pair][loc.name]

                    pat_port_i = graph.Ports.create(selector = port_i.selector,
                                                    port_io = 'in',
                                                    port_type= port_i.port_type,
                                                    neuron = neuron.name)
                    graph.Owns.create(int_0, pat_port_i)

                    pat_port_j = graph.Ports.create(selector = port_j.selector,
                                                    port_io = 'out',
                                                    port_type = port_j.port_type,
                                                    neuron = neuron.name)
                    graph.Owns.create(int_1, pat_port_j)

                    graph.SendsTo.create(pat_port_i, pat_port_j)
                    graph.SendsTo.create(port_i, pat_port_i)
                    graph.SendsTo.create(pat_port_j, port_j)

logger.info('Created Neuron Models and Ports')

for neuropil in neuropils:
    neuropil_name = neuropil.name
    lpu = lpu_dict[neuropil_name]['LPU']
    q_neuropil = query.QueryWrapper.from_objs(graph, [neuropil])
    synapses, _ = q_neuropil.traverse_owns(
                        max_levels = 1, cls = 'Synapse').get_as('obj')
    for synapse in synapses:
        exec_node = graph.AlphaSynapses.create(name = synapse.name)
        graph.Models.create(exec_node, synapse, version = model_version)
        graph.Owns.create(lpu, exec_node)
        q_synapse = query.QueryWrapper.from_objs(graph, [synapse])
        post_neuron, _ = q_synapse.gen_traversal_out(
                            ['SendsTo', 'Neuron', 'instanceof'],
                            min_depth = 1).gen_traversal_in(
                                ['Models', 'AxonHillockModel','instanceof'],
                                min_depth = 1).get_as('obj')
        graph.SendsTo.create(exec_node, post_neuron[0])

        q_pre_neuron = q_synapse.gen_traversal_in(
                            ['SendsTo', 'Neuron', 'instanceof'],
                            ['Models', 'AxonHillockModel','instanceof'],
                            min_depth = 2)
        pre_neuron = q_pre_neuron.get_as('obj')[0][0]
        pre_neuron_LPU = q_pre_neuron.owned_by(
                            cls = 'LPU', version = cx_version).get_as('obj')[0][0]
        if pre_neuron_LPU.name == neuropil_name:
            graph.SendsTo.create(pre_neuron, exec_node)
        else:
            q_port = q_pre_neuron.gen_traversal_out(
                        ['SendsTo', 'Port'],
                        ['SendsTo', 'Port'],
                        ['SendsTo', 'Port'],
                        ['SendsTo', 'Port'],
                        min_depth = 4)
            q_interface = q_port.owned_by(cls = 'Interface')
            df, _ = q_interface.get_as('df')
            rid = df.index[df['name'] == neuropil_name].tolist()
            if len(rid) > 1:
                raise ValueError("Multiple Ports Detected")
            else:
                port = (q_port+q_interface).edges_as_objs[0].inV()


            logger.info('created link between %s of %s: %s' % (
                            port.selector, pre_neuron.name, exec_node.name))
            graph.SendsTo.create(port, exec_node)

logger.info('Created Synapse Models')
