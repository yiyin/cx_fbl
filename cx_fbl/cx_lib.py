
import itertools
import json
from copy import deepcopy

import networkx as nx
import numpy as np

from .parse_arborization import NeuronArborizationParser
from .add_neuron import generate_svg

def na_to_nx(x):
    """Builds a NetworkX graph using processed data from NeuroArch.
    # Arguments:
        x (dict): File url.
    # Returns:
        g (NetworkX MultiDiGraph): A MultiDiGraph instance with the circuit graph.
    """
    g = nx.MultiDiGraph()
    g.add_nodes_from(x['nodes'].items())
    for pre,v,attrs in x['edges']:
        g.add_edge(pre, v, **attrs)
    return g

def nx_data_to_graph(data):
    g = nx.MultiDiGraph()
    g.add_nodes_from(list(data['nodes'].items()))
    for u, v, d in data['edges']:
        g.add_edge(u, v, **d)
    return g

class CX_Constructor(object):
    def __init__(self, fbl, initial_model_version):
        self.fbl = fbl
        self.initial_model_version = initial_model_version

        # CX_neuropils = ['PB','EB','BU','bu','FB','NO', 'no', 'LAL', 'lal', 'CRE', 'cre', \
        #                 'IB', 'ib', 'PS', 'ps', 'WED', 'wed']
        self.CX_LPUs = [
                        'PB','FB','EB',#'NO', 'no',
                        'BU','bu'#, 'LAL', 'lal', 'cre', 'CRE'
                        ]
        self.CX_Patterns = list(set(['-'.join(sorted(a)) for a in \
                                itertools.product(self.CX_LPUs, self.CX_LPUs)])\
                         - set(['-'.join(a) for a in \
                                zip(self.CX_LPUs, self.CX_LPUs)]))

        self._load_models()
        self.parser = NeuronArborizationParser()
        self._initialize_diagram_config()

    def na_query(self, query_list, format = 'nx'):
        inp = {"query": query_list,
               "format": format,
               "user": self.fbl.client._async_session._session_id,
               "server": self.fbl.naServerID}
        res = self.fbl.client.session.call('ffbo.processor.neuroarch_query',
                                           inp)
        data = res['success']['result']['data']
        return data

    def _load_models(self):
        query_list = [
                {"action": {"method":
                    {"query":{"name": self.CX_LPUs,
                              "version": self.initial_model_version}}},
                 "object":{"class":"LPU"}},
                {"action":{"method":
                    {"query":{"name": self.CX_Patterns,
                              "version": self.initial_model_version}}},
                 "object":{"class":"Pattern"}},
                {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                {"action": {"method":{"traverse_owns":{}}},
                 "object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}}
              ]
        data = self.na_query(query_list)
        self.models = nx_data_to_graph(data)
        # self.LPUs = {lpu: na_to_nx(v) for lpu, v in data['LPU'].items()}
        # self.Patterns = {p: na_to_nx(v) for p, v in data['Pattern'].items()}

    def _initialize_diagram_config(self):
        res_info = self.fbl.client.session.call(u'ffbo.processor.server_information')
        msg = {"user": self.fbl.client._async_session._session_id,
            "servers": {'na': self.fbl.naServerID, 'nk': list(res_info['nk'].keys())[0]}}
        res = self.fbl.client.session.call(u'ffbo.na.query.' + msg['servers']['na'], {'user': msg['user'],
                                'command': {"retrieve":{"state":0}},
                                'format': "nk"})
        newConfig = {'cx': {'disabled': []}}
        for lpu in res['data']['LPU'].keys():
            for node in res['data']['LPU'][lpu]['nodes']:
                if 'name' in res['data']['LPU'][lpu]['nodes'][node]:
                    node_data = res['data']['LPU'][lpu]['nodes'][node]
                    new_node_data = {'params': {},'states': {}}
                    for param in ['reset_potential', 'capacitance', 'resting_potential', 'resistance']:
                        if param in node_data:
                            new_node_data['params'][param] = node_data[param]
                            new_node_data['name'] = 'LeakyIAF'
                    for state in ['initV']:
                        if state in node_data:
                            new_node_data['states'][state] = node_data[state]
                    newConfig['cx'][res['data']['LPU'][lpu]['nodes'][node]['name']] = new_node_data

        newConfig_tosend = json.dumps(newConfig)
        self.fbl.JSCall(messageType='setExperimentConfig',data=newConfig_tosend)

    def show_removed_neurons(self):
        print(self.fbl.simExperimentConfig['cx']['disabled'])

    def remove_neurons(self):
        query_list = [{"action":{"method":{"has":{"name": self.fbl.simExperimentConfig['cx']['disabled']}}},"object":{"state":0}},
                {"action":{"method":{"get_connected_ports":{}}},"object":{"memory":0}},
                {"action":{"op":{"find_matching_ports_from_selector":{"memory":0}}},"object":{"state":0}},
                {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                {"action":{"method":{"gen_traversal_in":{"min_depth":1, "pass_through":["SendsTo", "SynapseModel","instanceof"]}}},"object":{"memory":0}},
                {"action":{"method":{"gen_traversal_out":{"min_depth":1, "pass_through":["SendsTo", "SynapseModel","instanceof"]}}},"object":{"memory":1}},
                {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":3}}},"object":{"memory":0}},
                {"action":{"op":{"__sub__":{"memory":0}}},"object":{"state":0}}]
        self.na_query(query_list)

    def prepare_execution(self):
        res_info = self.fbl.client.session.call(u'ffbo.processor.server_information')
        msg = {"user": self.fbl.client._async_session._session_id,
            "servers": {'na': self.fbl.naServerID, 'nk': list(res_info['nk'].keys())[0]}}
        res = self.fbl.client.session.call(u'ffbo.na.query.' + msg['servers']['na'], {'user': msg['user'],
                                'command': {"retrieve":{"state":0}},
                                'format': "nk"})
        return res

    def execute(self, res):
        self.fbl.execute_multilpu(res)

    def get_result(self):
        i = -2
        sim_output = json.loads(self.fbl.data[-1]['data']['data'])
        sim_output_new = json.loads(self.fbl.data[i]['data']['data'])
        while 'ydomain' in sim_output_new.keys():
            sim_output['data'].update(sim_output_new['data'])
            i = i - 1
            try:
                sim_output_new = json.loads(self.fbl.data[i]['data']['data'])
            except:
                break
        bs = []
        keys = []
        for key in sim_output['data'].keys():
            A = np.array(sim_output['data'][key])
            b = A[:,1]
            keys.append(key)
            bs.append(b)

        B = np.array(bs)
        print('Shape of Results:', B.shape)
        return B, keys
