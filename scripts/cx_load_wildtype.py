import csv
import json
import re
import sys
import pickle
import itertools
import logging

import path

from cx_fbl.parse_arborization import NeuronArborizationParser

import numpy as np
import numpy as np
from pyorient.ogm import Graph, Config

import neuroarch.models as models
import neuroarch.query as query
import neuroarch.nxtools as nxtools

from cx_config import cx_db, cx_version, initial_drop, model_version

def convert_set_to_list(a):
    for key in a:
        if isinstance(a[key],set):
            a[key] = list(a[key])
    return a

def convert_complicate_list(a):
    tmp = []
    for region in a['regions']:
        tmp.append(list(region))
    a['regions'] = tmp
    return a

# init db
# initial_drop = True

# test
# cx_db = 'localhost:8889/cx'
graph = Graph(Config.from_url(cx_db, 'root', 'root',
                              initial_drop=initial_drop))
# models.create_efficiently(graph, models.Node.registry)
# models.create_efficiently(graph, models.Relationship.registry)
graph.create_all(models.Node.registry)
graph.create_all(models.Relationship.registry)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('cx')

logger.info('Start to Load Biological Data')

CX_Neuropils = [
                'PB','FB','EB','NO', 'no',
                'BU','bu', 'LAL', 'lal', 'cre', 'CRE',
                # 'IB', 'ib', 'PS', 'ps', 'WED', 'wed'
                ]

neuropil_name_to_node = {}
for neuropil in CX_Neuropils:
    node = graph.Neuropils.create(name = neuropil, version = cx_version)
    neuropil_name_to_node[neuropil] = node

logger.info('Created Neuropils')

# File names grouped by neuropil in which neurons' presynaptic terminals
# arborize:
real_data = path.Path('real_neuron_data_new')
hypo_data = path.Path('hypo_neuron_data_new')

# FB subregions list
fb_subregions_list = []
for i in range(1,6):
    for j in range(1,5):
        fb_subregions_list.append('({},L{})'.format(i,j))
    for j in range(1,5):
        fb_subregions_list.append('({},R{})'.format(i,j))

# LAL subregions list
LAL_subregions_list = ['RGT', 'RDG', 'RVG', 'RHB']
# lal subregions list
lal_subregions_list = ['LGT', 'LDG', 'LVG', 'LHB']

# NO subregions list
NO_subregions_list = ['(1,R)', '(2,RD)', '(2,RV)','(3,RP)','(3,RM)','(3,RA)']
# no subregions list
no_subregions_list = ['(1,L)', '(2,LD)', '(2,LV)','(3,LP)','(3,LM)','(3,LA)']



neuropil_to_subregions = {'BU': ['R{}'.format(i) for i in range(1,81)],
                        'bu': ['L{}'.format(i) for i in range(1,81)],
                        'PB': ['L'+str(i) for i in range(1,10)]+['R'+str(i) for i in range(1,10)],
                        'EB': ['L'+str(i) for i in range(1,9)]+['R'+str(i) for i in range(1,9)],#+[str(i) for i in range(1,9)],
                        'FB': fb_subregions_list,
                        'LAL': LAL_subregions_list,
                        'lal': lal_subregions_list,
                        'NO': NO_subregions_list,
                        'no': no_subregions_list,
                        # 'IB': ['R'],
                        # 'ib': ['L'],
                        # 'PS': ['R'],
                        # 'ps': ['L'],
                        # 'WED': ['R'],
                        # 'wed': ['L'],
                        'CRE': ['RRB', 'RCRE'],
                        'cre': ['LRB', 'LCRE']}


neuropil_to_file_list = {'BU': hypo_data.files('bu_eb_1.csv'),
                         'bu': hypo_data.files('bu_eb_2.csv'),
                         'FB': real_data.files('fb_local.csv'), #+\
                         'EB': real_data.files('eb_lal_pb.csv'),
                         'PB': real_data.files('pb*.csv')#+\
                         # real_data.files('wed_ps_pb.csv')+\
                         # real_data.files('ib_lal_ps_pb.csv')
                        }

# File names grouped by neuron family:
family_to_file_list = {'BU-EB': hypo_data.files('bu_eb_*.csv'),
                       'FB': real_data.files('fb_local.csv'), #+\
                       #hypo_data.files('fb_local_*.csv'),
                       'EB-LAL-PB': real_data.files('eb_lal_pb.csv'),
                       # 'IB-LAL-PS-PB': real_data.files('ib_lal_ps_pb.csv'),
                       'PB-EB-LAL': real_data.files('pb_eb_lal.csv'),
                       'PB-EB-NO': real_data.files('pb_eb_no.csv'),
                       'PB-FB-CRE': real_data.files('pb_fb_cre.csv'),
                       'PB-FB-LAL': real_data.files('pb_fb_lal*.csv'),
                       'PB-FB-NO': real_data.files('pb_fb_no*.csv'),
                       'PB': real_data.files('pb_local.csv'),
                       # 'WED-PS-PB': real_data.files('wed_ps_pb.csv')
                       }

# Map file names to neuron family:
file_to_family = {}
for family in family_to_file_list:
    for file_name in family_to_file_list[family]:

        # Sanity check against typos in file list:
        if file_name in file_to_family:
            raise RuntimeError('duplicate file name')
        file_to_family[file_name] = family

# Parse labels into neuron data (i.e., each neuron associated with its label)
# and arborization data (i.e., each arborization associated with its originating
# label):
parser = NeuronArborizationParser()
# neuropil_data = [{'name': neuropil} for neuropil in neuropil_to_file_list]
# for retreiving node by name
subregion_name_to_node = {}

for neuropil in CX_Neuropils:
    for subregion in neuropil_to_subregions[neuropil]:
        node = graph.Subregions.create(name = '{}/{}'.format(neuropil, subregion))
        node.update(**{'neuropil': neuropil})
        subregion_name_to_node[frozenset([neuropil,subregion])] = node
        graph.Owns.create(neuropil_name_to_node[neuropil], node)

logger.info('Created Subregions')

neuron_name_to_node = {}
arbor_name_to_node = {}
terminal_name_to_node = {}
tract_dict = {}
neuron_rid_to_neuropil = {}

for neuropil in neuropil_to_file_list.keys():
    for file_name in neuropil_to_file_list[neuropil]:
        with open(file_name, 'r') as f:
            r = csv.reader(f, delimiter=' ')
            for row in r:
                d = {'name': row[0], 'family': file_to_family[file_name],
                     'neuropil': neuropil}

                # Add 'neuropil' attrib to each neuron data entry to enable ETL to
                # link each Neuron node to the appropriate Neuropil node:
#                 neuron_data.append(d)
                node = graph.Neurons.create(name = d['name'],
                                            family = file_to_family[file_name])
                neuron_name_to_node[d['name']] = node
                graph.Owns.create(neuropil_name_to_node[neuropil], node)
                neuron_rid_to_neuropil[node._id] = neuropil_name_to_node[neuropil]
                try:
                    tmp = parser.parse(row[0])
                except Exception as e:
                    print(file_name, row[0])
                    raise e

                axon_neuropils = set([a['neuropil'] for a in tmp \
                                  if 'b' in a['neurite']])
                # dendrite_neuropils = [a['neuropil'] for a in tmp \
                #                       if 's' in a['neurite']]
                # if len(dendrite_neuropils) > 1:
                #     raise ValueError(
                #         "Neuron {} has dendrite in more than one neuropil".format(node.name))
                # else:
                # dendrite_neuropil = dendrite_neuropils[0]
                dendrite_neuropil = neuropil
                for axon_neuropil in axon_neuropils:
                    if dendrite_neuropil != axon_neuropil:
                        neuropil_pair = frozenset(
                                        [dendrite_neuropil, axon_neuropil])
                        if neuropil_pair not in tract_dict:
                            tract_name = '-'.join(sorted(list(neuropil_pair)))
                            tract = graph.Tracts.create(name = tract_name,
                                                        version = cx_version,
                                                        neuropils = set(neuropil_pair))
                            tract_dict[neuropil_pair] = tract
                        graph.Owns.create(tract_dict[neuropil_pair], node)

                # Add 'neuron' attrib to each arborization data entry to enable
                # ETL to link each ArborizationDAta node to the appropriate
                # Neuron node:
                for a in tmp:
                    a['neuron'] = row[0]
#                     arbor_data.append(a)
                    node = graph.ArborizationDatas.create(name='arbor')
#                     print(a)
#                     print(type(a['regions']))
                    a = convert_set_to_list(a)
                    complicate_list = ['FB','NO','no']
                    if a['neuropil'] in complicate_list:
                        a = convert_complicate_list(a)
                    # node.update(**a)
                    # graph.Owns.create(neuron_name_to_node[a['neuron']], node)

                    for b in a['regions']:
                        # terminal = {}
                        # terminal['name'] = str(b).replace("'","")
                        # terminal['neurite'] = a['neurite']
                        # terminal['neuropil'] = a['neuropil']
                        # terminal['neuron'] = a['neuron']
                        # # terminal_data.append(terminal)
                        # node = graph.NeuronTerminals.create(name=terminal['name'])
                        # terminal = convert_set_to_list(terminal)
                        # node.update(**terminal)
                        # graph.Owns.create(neuron_name_to_node[a['neuron']], node)
                        if isinstance(b, list):
                            subregion_name = '({})'.format(','.join(b))
                        else:
                            subregion_name = b
                        graph.ArborizesIn.create(
                            neuron_name_to_node[a['neuron']],
                            subregion_name_to_node[frozenset([a['neuropil'],subregion_name])],
                            kind = a['neurite'])

logger.info('Created Neurons and Tracts')

q = query.QueryWrapper(graph,
                       query.QueryString(
                         "select from Neuropil where name in {} and version='{}'".format(
                              "['"+"','".join(CX_Neuropils)+"']", cx_version),
                         'sql'))
q_subregions = q.owns(1,cls="Subregion")
subregions, _ = q_subregions.get_as('obj')

for subregion in subregions:
    qq = query.QueryWrapper.from_objs(graph, [subregion])
    subregion_neuropil = qq.owned_by(
                            cls = 'Neuropil',
                            version = cx_version).get_as('obj')[0][0]
    g = qq.gen_traversal_in(['ArborizesIn', 'Neuron', 'instanceof'])
    df_nodes, df_edges = g.get_as('df')
    b = []
    s = []
    for ind in df_edges.index:
        neuron_rid = df_edges.loc[ind]['out']
        for kind in df_edges.loc[ind]['kind']:
            if kind == 's':
                s.append(neuron_rid)
            elif kind == 'b':
                b.append(neuron_rid)

    for post_rid in s:
        post_neuropil = neuron_rid_to_neuropil[post_rid]
        for pre_rid in b:
            if pre_rid != post_rid:
                name = df_nodes.loc[pre_rid]['name']+'->'+ \
                       df_nodes.loc[post_rid]['name']+'_in_'+ \
                       subregion.name.replace('/','-')
                syn = graph.Synapses.create(name = name)

                graph.SendsTo.create(graph.get_element(pre_rid), syn)
                graph.SendsTo.create(syn, graph.get_element(post_rid))
                graph.Owns.create(subregion, syn)
                graph.Owns.create(post_neuropil, syn)
                if subregion_neuropil.name != post_neuropil.name:
                    pre_neuropil = neuron_rid_to_neuropil[pre_rid]
                    if pre_neuropil.name != post_neuropil.name:
                        neuropil_pair = frozenset([pre_neuropil.name, post_neuropil.name])
                        if neuropil_pair not in tract_dict:
                            tract_name = '-'.join(sorted(list(neuropil_pair)))
                            tract = graph.Tracts.create(name = tract_name,
                                                        version = cx_version,
                                                        neuropils = set(neuropil_pair))
                            tract_dict[neuropil_pair] = tract
                            logger.info('created new tract {}: {}'.format(tract.name, syn.name))
                        graph.Owns.create(tract_dict[neuropil_pair],
                                          graph.get_element(pre_rid))

logger.info('Created Synapses')

logger.info('Biological Data Loaded')
