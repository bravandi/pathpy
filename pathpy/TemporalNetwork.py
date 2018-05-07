# -*- coding: utf-8 -*-
"""
    pathpy is an OpenSource python package for the analysis of sequential data on pathways and temporal networks using higher- and multi order graphical models

    Copyright (C) 2016-2017 Ingo Scholtes, ETH Zürich

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Contact the developer:
    
    E-mail: ischoltes@ethz.ch
    Web:    http://www.ingoscholtes.net
"""

import numpy as _np
import sys as _sys
import collections as _co
import bisect as _bs
import datetime as _dt
import time as _t

from pathpy.Log import Log
from pathpy.Log import Severity
import pathpy.Paths
import networkx as nx
import networkx.algorithms.flow as nx_flow


class TemporalNetwork:
    """ This class represents a sequence of time-stamped edges.
       Instances of this class can be used to generate path statistics 
       based on the time-respecting paths resulting from a given maximum
       time difference between consecutive time-stamped edges.
    """

    def __init__(self, tedges=None):
        """
        Constructor that generates a temporal network instance. 
        
        @param tedges: an optional list of (possibly unordered time-stamped) links 
            from which to construct a temporal network instance. For the default value None        
            an empty temporal network will be created.
        """

        ## A list of time-stamped edges of this temporal network
        self.tedges = []

        ## A list of nodes of this temporal network
        self.nodes = []

        ## A dictionary storing all time-stamped links, indexed by time-stamps
        self.time = _co.defaultdict(lambda: list())

        ## A dictionary storing all time-stamped links, indexed by time and target node
        self.targets = _co.defaultdict(lambda: dict())

        ## A dictionary storing all time-stamped links, indexed by time and source node 
        self.sources = _co.defaultdict(lambda: dict())

        ## A dictionary storing time stamps at which links (v,*;t) originate from node v
        self.activities = _co.defaultdict(lambda: list())

        ## A dictionary storing sets of time stamps at which links (v,*;t) originate from node v
        ## Note that the insertion into a set is much faster than repeatedly checking whether 
        ## an element already exists in a list!
        self.activities_sets = _co.defaultdict(lambda: set())

        ## An ordered list of time-stamps
        self.ordered_times = []

        nodes_seen = _co.defaultdict(lambda: False)

        if tedges is not None:
            Log.add('Building index data structures ...')

            for e in tedges:
                self.activities_sets[e[0]].add(e[2])
                self.time[e[2]].append(e)
                self.targets[e[2]].setdefault(e[1], []).append(e)
                self.sources[e[2]].setdefault(e[0], []).append(e)
                if not nodes_seen[e[0]]:
                    nodes_seen[e[0]] = True
                if not nodes_seen[e[1]]:
                    nodes_seen[e[1]] = True
            self.tedges = tedges
            self.nodes = list(nodes_seen.keys())

            Log.add('Sorting time stamps ...')

            self.ordered_times = sorted(self.time.keys())
            for v in self.nodes:
                self.activities[v] = sorted(self.activities_sets[v])
            Log.add('finished.')

    @staticmethod
    def readFile(filename, sep=',', timestampformat="%Y-%m-%d %H:%M", maxlines=_sys.maxsize):
        """ Reads time-stamped links from a file and returns a new instance 
            of the class TemporalNetwork. The file is assumed to have a header 

                source target time 

            where columns can be in arbitrary order and separated by arbitrary characters. 
            Each time-stamped link must occur in a separate line and links are assumed to be
            directed.
             
            The time column can be omitted and in this case all links are assumed to occur 
            in consecutive time stamps (that have a distance of one). Time stamps can be simple 
            integers, or strings to be converted to UNIX time stamps via a custom timestamp format. 
            For this, the python function datetime.strptime will be used. 

            @param sep: the character that separates columns 
            @param filename: path of the file to read from
            @param timestampformat: used to convert string timestamps to UNIX timestamps. This parameter is 
                ignored, if the timestamps are digit types (like a simple int).
            @param maxlines: limit reading of file to certain number of lines, default sys.maxsize

        """
        assert (filename != ''), 'Empty filename given'

        # Read header
        with open(filename, 'r') as f:
            tedges = []

            header = f.readline()
            header = header.split(sep)

            # If header columns are included, arbitrary column orders are supported
            time_ix = -1
            source_ix = -1
            mid_ix = -1
            weight_ix = -1
            target_ix = -1
            for i in range(len(header)):
                header[i] = header[i].strip()
                if header[i] == 'node1' or header[i] == 'source':
                    source_ix = i
                elif header[i] == 'node2' or header[i] == 'target':
                    target_ix = i
                elif header[i] == 'time' or header[i] == 'timestamp':
                    time_ix = i

            assert (source_ix >= 0 and target_ix >= 0), "Detected invalid header columns: %s" % header

            if time_ix < 0:
                Log.add('No time stamps found in data, assuming consecutive links', Severity.WARNING)

            Log.add('Reading time-stamped links ...')

            line = f.readline()
            n = 1
            while line and n <= maxlines:
                fields = line.rstrip().split(sep)
                try:
                    if time_ix >= 0:
                        timestamp = fields[time_ix]
                        # if the timestamp is a number, we use this 
                        if timestamp.isdigit():
                            t = int(timestamp)
                        else:  # if it is a string, we use the timestamp format to convert it to a UNIX timestamp
                            x = _dt.datetime.strptime(timestamp, timestampformat)
                            t = int(_t.mktime(x.timetuple()))
                    else:
                        t = n
                    if t >= 0:
                        tedge = (fields[source_ix], fields[target_ix], t)
                        tedges.append(tedge)
                    else:
                        Log.add('Ignoring negative timestamp in line ' + str(n + 1) + ': "' + line.strip() + '"',
                                Severity.WARNING)
                except (IndexError, ValueError):
                    Log.add('Ignoring malformed data in line ' + str(n + 1) + ': "' + line.strip() + '"',
                            Severity.WARNING)
                line = f.readline()
                n += 1
        # end of with open()

        return TemporalNetwork(tedges=tedges)

    def filterEdges(self, edge_filter):
        """Filter time-stamped edges according to a given filter expression. 

        @param edge_filter: an arbitrary filter function of the form filter_func(v, w, time) that 
            returns True for time-stamped edges that shall pass the filter, and False for all edges that 
            shall be filtered out.
        """

        Log.add('Starting filtering ...', Severity.INFO)
        new_t_edges = []

        for (v, w, t) in self.tedges:
            if edge_filter(v, w, t):
                new_t_edges.append((v, w, t))

        Log.add('finished. Filtered out ' + str(self.ecount() - len(new_t_edges)) + ' time-stamped edges.',
                Severity.INFO)

        return TemporalNetwork(tedges=new_t_edges)

    def convertStaticNetwork(self, normalzie=False):
        """
        :param normalzie:   Map node names into a sequence of integers starts from 0 to len(nodes) - 1. If set True the
                            return dictinary wont have the 'dict_source_mapped' and 'dict_mapped_source' keys.

        Converts the network to its static representation where the weight of edges represent
        the frequency of that edge occurance with the observation period.
        The normalizition process maps each node to an integer in range(0, number of nodes) and returns the list of edges
        :return: {
                    'dict_source_mapped': dict[node]=mapped index,
                    'dict_mapped_source': dict[mapped index]=node,
                    'edges_weight_frequency': [(from, to, frequency)]
                    'edges_frequency_dict': dict[(from, to)] = frequency
                }
        """

        edges_frequency_dict = dict()
        edges_weight_frequency = []

        if normalzie is False:

            for edge in self.tedges:
                edge_key = (edge[0], edge[1])

                if edge_key in edges_frequency_dict:
                    edges_frequency_dict[edge_key] += 1
                else:
                    edges_frequency_dict[edge_key] = 1

            for edge_key, frequency in edges_frequency_dict.items():
                edges_weight_frequency.append(edge_key + (frequency,))

            return {
                'edges_weight_frequency': edges_weight_frequency,
                'edges_frequency_dict': edges_frequency_dict
            }

        else:

            node_index = 0
            dict_source_mapped = dict()
            dict_mapped_source = dict()

            for node in self.nodes:
                dict_source_mapped[node] = node_index
                dict_mapped_source[node_index] = node

                node_index = node_index + 1

            for edge in self.tedges:
                edge_key = (dict_source_mapped[edge[0]], dict_source_mapped[edge[1]])

                if edge_key in edges_frequency_dict:
                    edges_frequency_dict[edge_key] += 1
                else:
                    edges_frequency_dict[edge_key] = 1

            edges_weight_frequency = []

            for edge_key, frequency in edges_frequency_dict.items():
                edges_weight_frequency.append(edge_key + (frequency,))

            return {
                'dict_source_mapped': dict_source_mapped,
                'dict_mapped_source': dict_mapped_source,
                'edges_weight_frequency': edges_weight_frequency,
                'edges_frequency_dict': edges_frequency_dict
            }

        pass

    def snapshots_observation_period(self, interval=50, delta_t_list=None):
        """
        Create snapshots within the interval or the given delta times.
        The comparsion is of intervals is [x, x + delta)
        :param      interval: will create the delta_t_list from range(0, max(timestamp), interval). Default=50
        :param      delta_t_list: Optional. If given the interval will be ignored.
                    List containing tuples of delta-times (>=, <) to create snapshots from
        :return:    List of TemporalNetworks with all edges ts=0
        """

        if delta_t_list is None:
            rng = [r for r in range(0, self.ordered_times[-1], interval)]

            # make sure last element is in the list and +1 because the intervals are [,)
            if rng[-1] != self.ordered_times[-1]:
                rng.append(self.ordered_times[-1] + 1)
                pass
            else:
                rng[-1] = rng[-1] + 1
                pass

            delta_t_list = []
            start = 0
            for r in rng[1:]:  # dont include 0 twice
                delta_t_list.append((start, r))
                start = r

            pass
        else:
            if type(delta_t_list) == tuple:
                delta_t_list = [delta_t_list]

        result = dict()

        times = _np.array(list(self.targets.keys()))

        for delta in delta_t_list:

            # query times for faster retrieval since it is a sorted list
            edges = _np.where(_np.logical_and(times >= delta[0], times < delta[1]))

            if len(edges[0]) > 0:
                tn = TemporalNetwork()
                result[delta] = tn

                for time_index in edges[0]:
                    for vertex, vertex_edge in self.targets[times[time_index]].items():
                        for edge in vertex_edge:
                            tn.addEdge(source=edge[0], target=edge[1], ts=0)  # ts=edge[2]

                    # print("{}: {}".format(times[time_index], self.targets[times[time_index]]))

                    pass
            else:
                result[delta] = None

        return result
        pass

    def snapshots_continues(self, delta_t, stop_if_static_graph_is_connected=True):
        """
        Continusly creates aggreagted M snapshots with respect to delta_t.
        This is same as adding an increasing sequence of snapshots observing from 0 to M * delta_t time.
        For example, snapshot[1] = snapshot[0] + snapshot[1] = all links in observation period [0, 2 * delta_t].
        :param      delta_t: time gap to build snapshots
        :param      stop_if_static_graph_is_connected: stop the process if the static represntation of graph becomes an
                    undirected connected graph
        :return:    dictionary of TemporalNetworks key=observation period
        """

        result = dict()

        for obs_period in range(delta_t, self.ordered_times[-1] + 1, delta_t):
            r = self.snapshots_observation_period(delta_t_list=(0, obs_period))

            r_key = list(r.keys())[0]

            r_t_net = r[r_key]
            result[r_key] = r_t_net

            if stop_if_static_graph_is_connected is True:
                G = nx.Graph()

                static_rep = r_t_net.convertStaticNetwork(normalzie=False)

                G.add_edges_from(static_rep["edges_frequency_dict"].keys())

                if nx.is_connected(G) is True:
                    break
            pass

        return result

    def addNode(self, new_node):
        """

        :param new_node:
        :return:
        """

        if new_node not in self.nodes:
            self.nodes.append(new_node)

    def addEdge(self, source, target, ts):
        """Adds a directed time-stamped edge (source,target;time) to the temporal network. To add an undirected 
            time-stamped link (u,v;t) at time t, please call addEdge(u,v;t) and addEdge(v,u;t).
        
        @param source: name of the source node of a directed, time-stamped link
        @param target: name of the target node of a directed, time-stamped link
        @param ts: (integer) time-stamp of the time-stamped link
        """
        e = (source, target, ts)
        self.tedges.append(e)
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)

        # Add edge to index structures
        self.time[ts].append(e)
        self.targets[ts].setdefault(target, []).append(e)
        self.sources[ts].setdefault(source, []).append(e)

        if ts not in self.activities[source]:
            self.activities[source].append(ts)
            self.activities[source].sort()

        # Reorder time stamps
        self.ordered_times = sorted(self.time.keys())

    def removeEdge(self, e):
        """

        :param e:
        :return:
        """
        occurances = self.tedges.count(e)
        if occurances > 1:
            raise Exception("Edge {} is occured {} times.".format(e, occurances))
        if occurances == 0:
            return

        self.tedges.remove(e)

        source = e[0]
        target = e[1]
        ts = e[2]

        # Add edge to index structures
        self.time[ts].remove(e)
        self.targets[ts][target].remove(e)
        self.sources[ts][source].remove(e)

        if ts in self.activities[source]:
            # Depends how self.activities is used. If the order of items is important
            # this removal can invalidate the datastructure, in case two different edges
            # happens in the exact same time.

            self.activities[source].remove(ts)
            self.activities[source].sort()

        # Reorder time stamps
        self.ordered_times = sorted(self.time.keys())

    def vcount(self):
        """
        Returns the number of vertices in the temporal network. 
        This number corresponds to the number of nodes in the (first-order) 
        time-aggregated network.
        """

        return len(self.nodes)

    def ecount(self):
        """
        Returns the number of time-stamped edges (u,v;t) in the temporal network.
        This number corresponds to the sum of link weights in the (first-order)
        time-aggregated network.
        """

        return len(self.tedges)

    def getObservationLength(self):
        """
        Returns the length of the observation time in time units.
        """

        return max(self.ordered_times) - min(self.ordered_times)

    def getInterEventTimes(self):
        """
        Returns an array containing all time differences between any 
        two consecutive time-stamped links (involving any node)
        """

        timediffs = []
        for i in range(1, len(self.ordered_times)):
            timediffs += [self.ordered_times[i] - self.ordered_times[i - 1]]
        return _np.array(timediffs)

    def getInterPathTimes(self):
        """
        Returns a dictionary which, for each node v, contains all time differences 
        between any time-stamped link (*,v;t) and the next link (v,*;t') (t'>t)
        in the temporal network
        """

        interPathTimes = _co.defaultdict(lambda: list())
        for e in self.tedges:
            # Get target v of current edge e=(u,v,t)
            v = e[1]
            t = e[2]

            # Get time stamp of link (v,*,t_next) with smallest t_next such that t_next > t
            i = _bs.bisect_right(self.activities[v], t)
            if i != len(self.activities[v]):
                interPathTimes[v].append(self.activities[v][i] - t)
        return interPathTimes

    def summary(self):
        """
        Returns a string containing basic summary statistics of this temporal network
        """

        summary = ''

        summary += 'Nodes:\t\t\t' + str(self.vcount()) + '\n'
        summary += 'Time-stamped links:\t' + str(self.ecount()) + '\n'
        if self.vcount() > 0:
            summary += 'Links/Nodes:\t\t' + str(self.ecount() / self.vcount()) + '\n'
        else:
            summary += 'Links/Nodes:\t\tN/A\n'
        if len(self.ordered_times) > 1:
            summary += 'Observation period:\t[' + str(min(self.ordered_times)) + ', ' + str(
                max(self.ordered_times)) + ']\n'
            summary += 'Observation length:\t' + str(max(self.ordered_times) - min(self.ordered_times)) + '\n'
            summary += 'Time stamps:\t\t' + str(len(self.ordered_times)) + '\n'

            d = self.getInterEventTimes()
            summary += 'Avg. inter-event dt:\t' + str(_np.mean(d)) + '\n'
            summary += 'Min/Max inter-event dt:\t' + str(min(d)) + '/' + str(max(d)) + '\n'

        return summary

    def __str__(self):
        """
        Returns the default string representation of 
        this temporal network instance.
        """
        return self.summary()

    def copy(self):
        """

        :return:
        """
        t = TemporalNetwork()

        for node in self.nodes:
            t.addNode(node)

        for edge in self.tedges:
            t.addEdge(edge[0], edge[1], edge[2])

        return t

    def ShuffleEdges(self, l=0, with_replacement=False):
        """
        Generates a shuffled version of the temporal network in which edge statistics (i.e.
        the frequencies of time-stamped edges) are preserved, while all order correlations are 
        destroyed. The shuffling procedure randomly reshuffles the time-stamps of links.
        
        @param l: the length of the sequence to be generated (i.e. the number of time-stamped links.
            For the default value l=0, the length of the generated shuffled temporal network will be 
            equal to that of the original temporal network. 
        @param with_replacement: Whether or not the sampling should be with replacement (default False)
        """

        tedges = []

        timestamps = [e[2] for e in self.tedges]
        edges = list(self.tedges)

        if l == 0:
            l = len(self.tedges)
        for i in range(l):

            if with_replacement:
                # Pick random link
                edge = edges[_np.random.randint(0, len(edges))]
                # Pick random time stamp
                time = timestamps[_np.random.randint(0, len(timestamps))]
            else:
                # Pick random link
                edge = edges.pop(_np.random.randint(0, len(edges)))
                # Pick random time stamp
                time = timestamps.pop(_np.random.randint(0, len(timestamps)))

                # Generate new time-stamped link
            tedges.append((edge[0], edge[1], time))

        # Generate temporal network
        t = TemporalNetwork(tedges=tedges)

        # Fix node order to correspond to original network
        t.nodes = self.nodes

        return t

    def convertTimeUnfoldedNetworkx(self, memory=0, y_distance=1.5, layout=False):
        """

        :param memory: dict specify how long a node keeps information. If an integer is given
                       it will be assigned for all nodes. Default is infinity set by 0
        :param layout:
        :return:
        """

        if memory is not dict:
            memory_int = memory
            memory = dict()

            for node in self.nodes:
                memory[node] = memory_int
            del memory_int
        else:
            for node in self.nodes:
                if node not in memory:
                    memory[node] = 0
            pass

        g = nx.DiGraph()

        sorted_nodes = _np.sort(self.nodes)
        prev_time = 0
        prev_time_nodes = []

        y_pos = y_distance

        i = 0
        for n in sorted_nodes:
            new_node = "0_{}".format(n)
            prev_time_nodes.append(new_node)

            if layout:
                g.add_node(new_node, pos="{},{}!".format(i, 0), label=str(n))
                if i == 0:
                    g.node[new_node]["xlabel"] = "t=0"
            else:
                g.add_node(new_node)

            i += 1
            pass

        prev_ts = 0

        for ts in self.ordered_times:
            prev_time_nodes_temp = []

            i = 0
            for n in sorted_nodes:
                new_node = "{}_{}".format(ts, n)
                prev_time_nodes_temp.append(new_node)

                if layout:
                    g.add_node(new_node, pos="{},{}!".format(i, y_pos * -1), label=str(n))
                    if i == 0:
                        g.node[new_node]["xlabel"] = "t={}".format(ts)
                else:
                    g.add_node(new_node)

                if memory[n] == 0 or ts - prev_ts <= memory[n]:
                    if layout:
                        g.add_edge(prev_time_nodes[i], new_node, style="dotted")
                    else:
                        g.add_edge(prev_time_nodes[i], new_node)

                i += 1
                pass

            for edge in self.time[ts]:
                from_node = "{}_{}".format(prev_ts, edge[0])
                to_node = "{}_{}".format(edge[2], edge[1])

                if from_node not in g or to_node not in g:
                    raise Exception("Node does not exists.")

                mid_node = None

                if layout:
                    g.add_edge(
                        "{}_{}".format(prev_ts, edge[0]),
                        "{}_{}".format(edge[2], edge[1]),
                        style="solid")
                else:

                    g.add_edge(
                        "{}_{}".format(prev_ts, edge[0]),
                        "{}_{}".format(edge[2], edge[1]))
                pass

            del prev_time_nodes
            prev_time_nodes = prev_time_nodes_temp
            y_pos += y_distance
            prev_ts = ts
            pass

        return g
        pass

    def exportUnfoldedNetworkLatex(self, filename):
        """
        Generates a tex file that can be compiled to a time-unfolded 
        representation of the temporal network.

        @param filename: the name of the tex file to be generated.
        """

        import os as _os

        output = []

        output.append('\\documentclass{article}\n')
        output.append('\\usepackage{tikz}\n')
        output.append('\\usepackage{verbatim}\n')
        output.append('\\usepackage[active,tightpage]{preview}\n')
        output.append('\\PreviewEnvironment{tikzpicture}\n')
        output.append('\\setlength\PreviewBorder{5pt}%\n')
        output.append('\\usetikzlibrary{arrows}\n')
        output.append('\\usetikzlibrary{positioning}\n')
        output.append('\\begin{document}\n')
        output.append('\\begin{center}\n')
        output.append('\\newcounter{a}\n')
        output.append("\\begin{tikzpicture}[->,>=stealth',auto,scale=0.5, every node/.style={scale=0.9}]\n")
        output.append("\\tikzstyle{node} = [fill=lightgray,text=black,circle]\n")
        output.append("\\tikzstyle{v} = [fill=black,text=white,circle]\n")
        output.append("\\tikzstyle{dst} = [fill=lightgray,text=black,circle]\n")
        # output.append("\\tikzstyle{lbl} = [fill=white,text=black,circle]\n")
        output.append("\\tikzstyle{lbl} = [fill=white,text=black]\n")

        last = ''

        sorted_nodes = _np.sort(self.nodes)

        extended_ordered_times = self.ordered_times[:]
        extended_ordered_times.append(self.ordered_times[-1] + 1)

        for n in sorted_nodes:
            if last == '':
                output.append("\\node[lbl]                     (" + str(n) + "-0)   {$" + str(n) + "$};\n")
            else:
                output.append(
                    "\\node[lbl,right=0.5cm of " + str(last) + "-0] (" + str(n) + "-0)   {$" + str(n) + "$};\n")
            last = n

        output.append("\\setcounter{a}{0}\n")

        if False:
            output.append("\\foreach \\number in {" + str(min(self.ordered_times)) + ",...," + str(
                max(self.ordered_times) + 1) + "}{\n")
            output.append("\\setcounter{a}{\\number}\n")
            output.append("\\addtocounter{a}{-1}\n")
            output.append("\\pgfmathparse{\\thea}\n")

            for n in _np.sort(self.nodes):
                output.append(
                    "\\node[v,below=0.3cm of " + str(n) + "-\\pgfmathresult]     (" + str(n) + "-\\number) {};\n")
            output.append("\\node[lbl,left=0.5cm of " + str(_np.sort(self.nodes)[0]) +
                          "-\\number]    (col-\\pgfmathresult) {$t=$\\number};\n")
            output.append("}\n")
            pass

        prev_time = 0
        for time in extended_ordered_times:
            for n in sorted_nodes:
                # output.append("\\node[v,below=0.3cm of " + str(n) + "-\\pgfmathresult]     (" + str(n) + "-\\number) {};\n")
                output.append("\\node[v,below=0.3cm of {}-{}]     ({}-{}) {{}};\n".format(
                    n, prev_time, n, time
                ))
                pass
            output.append(
                "\\node[lbl,left=0.5cm of {}-{}]    (col-{}) {{$t=${}}};\n".format(sorted_nodes[0], time, prev_time,
                                                                                   time))
            prev_time = time

        output.append("\\path[->,thick]\n")
        i = 1

        time_index = 0
        for ts in extended_ordered_times[:-1]:
            for edge in self.time[ts]:
                if False:
                    output.append(
                        "(" + str(edge[0]) + "-" + str(ts) + ") edge (" + str(edge[1]) + "-" + str(ts + 1) + ")\n")
                    i += 1
                    pass

                if len(extended_ordered_times) == time_index + 1:
                    raise Exception("Time does not exists")

                output.append(
                    "(" + str(edge[0]) + "-" + str(ts) + ") edge (" + str(edge[1]) + "-" + str(
                        extended_ordered_times[time_index + 1]) + ")\n")
                pass
            time_index += 1
            pass

        output.append(";\n")
        output.append(
            """\end{tikzpicture}
            \end{center}
            \end{document}""")

        # create directory if necessary to avoid IO errors
        directory = _os.path.dirname(filename)
        if directory != '':
            if not _os.path.exists(directory):
                _os.makedirs(directory)

        with open(filename, "w") as tex_file:
            tex_file.write(''.join(output))

    def unfoldedNetworkControlMaxFlow(
            self, allowed_drivers=[], memory=0, stimuli_allowed_periods=[], layout=False,
            layout_include_flow_regulatory_nodes=False, force_shortest_path=False,
            find_all_source_to_sink_paths=True, color_set=None, find_max_capacity_each_source=False):
        """
        Generates a dot file that can be compiled to a time-unfolded
        representation of the temporal network.

        @param allowed_drivers: Fix
        @param memory:  dict specify how long a node keeps information. If an integer is given
                        it will be assigned for all nodes. Default is infinity set by -1
        @param layout: Fix
        @param layout_include_flow_regulatory_nodes: Fix
        @param find_all_source_to_sink_paths:
        @param color_set: a function that gets number of required distinct colors and returns an
                          array of colors
        """

        y_pos_distance = 1.5

        if len(stimuli_allowed_periods) == 0:
            """
            directly stimulating nodes in the last time does not make sense because the stimulation
            wont be carried to the next time (there is no next time).
            """
            stimuli_allowed_periods.append((0, self.ordered_times[-1]))

        sorted_nodes = sorted(self.nodes)

        if len(allowed_drivers) == 0:
            allowed_drivers = sorted_nodes
        else:
            allowed_drivers = sorted(allowed_drivers)

        unfolded_dnx = self.convertTimeUnfoldedNetworkx(memory=memory, layout=layout, y_distance=y_pos_distance)

        for node in allowed_drivers:
            if node not in self.nodes:
                raise Exception("Source/driver node does not exists. Node: {}".format(node))
            pass

        """
        Add an extra node for each node with in-deg > 1 and split in-deg and out-deg
        between this new node to make sure the residual capacity of all nodes is 1                
        """

        g_copy = unfolded_dnx.copy()
        for out_node, out_deg in g_copy.out_degree:
            # if out_deg > 1:
            if out_deg > 0:
                # [g[in_edge[0]][in_edge[1]] for in_edge in g.in_edges(node)]
                new_node = "{}$".format(out_node)

                if layout:
                    positions = unfolded_dnx.node[out_node]["pos"][:-1].split(",")
                    node_positions = (float(positions[0]), float(positions[1]))

                    unfolded_dnx.add_node(
                        new_node,
                        pos="{},{}!".format(node_positions[0], node_positions[1] - (y_pos_distance / 2) - 0.05)
                        if layout_include_flow_regulatory_nodes else
                        "{},{}!".format(node_positions[0], node_positions[1]),
                        label=out_node.split("_")[1] + "'" if layout_include_flow_regulatory_nodes else "",
                        style="" if layout_include_flow_regulatory_nodes else "invis"
                    )

                    unfolded_dnx.add_edge(
                        out_node, new_node,
                        style="" if layout_include_flow_regulatory_nodes else "invis",
                        capacity=1.0
                    )
                else:
                    unfolded_dnx.add_node(new_node)

                    unfolded_dnx.add_edge(out_node, new_node)

                for out_edge in g_copy.out_edges(out_node):
                    if layout:
                        if "style" in unfolded_dnx[out_edge[0]][out_edge[1]]:
                            unfolded_dnx.add_edge(
                                new_node, out_edge[1],
                                style=unfolded_dnx[out_edge[0]][out_edge[1]]["style"]

                            )
                    else:
                        unfolded_dnx.add_edge(new_node, out_edge[1])

                    unfolded_dnx.remove_edge(out_edge[0], out_edge[1])
                    pass
        del g_copy

        """
        set all edges capacity to 1 to find control paths and time unfolded driver nodes
        **
        """
        nx.set_edge_attributes(unfolded_dnx, 1.0, "capacity")
        # for edge in unfolded_dnx.edges:
        #     if "capacity" not in unfolded_dnx[edge[0]][edge[1]]:
        #
        #         unfolded_dnx[edge[0]][edge[1]]["capacity"] = len(self.nodes)
        # nx.set_edge_attributes(unfolded_dnx, len(self.nodes), "capacity")

        """
        Create Source node to initial stimulating time unfolded nodes
        within the Stimuli Allowed Period 
        """

        source_node = "s"

        if layout:
            unfolded_dnx.add_node(source_node, pos="{},{}!".format(
                len(self.nodes) + 2, -1 * len(self.ordered_times) / 2))
            positions = unfolded_dnx.node[source_node]["pos"][:-1].split(",")
            node_positions = (float(positions[0]), float(positions[1]))
        else:
            unfolded_dnx.add_node(source_node)

        y_val = -1
        for node in sorted_nodes:
            new_node = "s_{}".format(node)
            if layout:
                unfolded_dnx.add_node(new_node, pos="{},{}!".format(node_positions[0] - 1.5, y_val))
                y_val += -1
            else:
                unfolded_dnx.add_node(new_node)
                pass
            unfolded_dnx.add_edge(source_node, new_node, capacity=len(self.nodes))

        for node in unfolded_dnx.nodes:
            if node.startswith("s"):
                continue

            for source in allowed_drivers:
                node_ts = int(node.split("_")[0])
                allow_stimuli = False

                for stimuli_allowed_period in stimuli_allowed_periods:
                    if stimuli_allowed_period[0] <= node_ts < stimuli_allowed_period[1]:
                        allow_stimuli = True
                        break

                if allow_stimuli and node.endswith("_{}".format(source)):
                    super_source_node = "{}_{}".format(source_node, source)

                    if layout:
                        unfolded_dnx.add_edge(super_source_node, node, style="dashed")
                    else:
                        unfolded_dnx.add_edge(super_source_node, node)

                    unfolded_dnx[super_source_node][node]["capacity"] = 1.0
            pass

        """
        ############# Create sink node and edges
        """

        sink_node = "t"

        if layout:
            unfolded_dnx.add_node(sink_node, pos="{},{}!".format(
                (len(self.nodes) / 2) - 0.5, -1 * (len(self.ordered_times) + 2 + y_pos_distance)))
        else:
            unfolded_dnx.add_node(sink_node)

        for node in self.nodes:
            from_node = "{}_{}".format(self.ordered_times[-1], node)
            unfolded_dnx.add_edge(from_node, sink_node)

            # Sink edges capacity
            # **
            unfolded_dnx[from_node][sink_node]["capacity"] = 1.0
            # unfolded_dnx[from_node][sink_node]["capacity"] = len(self.nodes)
            pass

        # TODO Organize later
        if find_max_capacity_each_source is False:
            if force_shortest_path:
                flow_value, flow_dict = nx_flow.maximum_flow(
                    unfolded_dnx, source_node, sink_node,
                    flow_func=nx_flow.shortest_augmenting_path
                )
            else:
                flow_value, flow_dict = nx_flow.maximum_flow(
                    unfolded_dnx, source_node, sink_node)

        """
        find_max_capacity_each_source 
        """

        if find_max_capacity_each_source:
            print("--------Running Max Capacity Finder ---------\n"
                  "{}\n".format(self.summary()))

            super_source_max_capacity = dict()

            reverse_unfolded_dnx = unfolded_dnx.copy()
            for edge in unfolded_dnx.edges:
                reverse_unfolded_dnx.remove_edge(edge[0], edge[1])
                reverse_unfolded_dnx.add_edge(edge[1], edge[0], capacity=1)

            for node in self.nodes:
                super_source_node = "{}_{}".format(source_node, node)
                super_source_max_capacity[super_source_node] = {
                    "flow_value": None, "flow_dict": None
                }

                flow_value_to_source, flow_dict_to_source = nx_flow.maximum_flow(
                    reverse_unfolded_dnx, sink_node, super_source_node)

                del flow_dict_to_source
                super_source_max_capacity[super_source_node]["flow_value"] = flow_value_to_source
                super_source_max_capacity[super_source_node]["flow_dict"] = None

                print("-------------------\n"
                      "Number of super sources left: {}\n"
                      "For super source: {}\n"
                      "{}".format(
                    len(self.nodes) - len(super_source_max_capacity),
                    super_source_node,
                    super_source_max_capacity[super_source_node]
                ))
                pass

            del reverse_unfolded_dnx
            return super_source_max_capacity
            pass

        # unfolded_dnx = res
        """
        Remove in-active stimuli edges from the time-unfolded network 
        """

        if True:
            for node in sorted_nodes:
                super_source_node = "{}_{}".format(source_node, node)
                for origin, flow in flow_dict[super_source_node].items():
                    if unfolded_dnx.has_edge(super_source_node, origin) and flow == 0:
                        unfolded_dnx.remove_edge(super_source_node, origin)
                        pass

        """
        Default color and labels for control edges 
        """
        if layout:
            for from_node, to_nodes in flow_dict.items():
                for to_node, flow in to_nodes.items():
                    if flow > 0:
                        # default color for an edge with flow > 0
                        unfolded_dnx[from_node][to_node]["color"] = "#B900CA"
                        if from_node == source_node:
                            # unfolded_dnx[from_node][to_node]["label"] = "s>{}".format(to_node)
                            pass
                        elif "style" in unfolded_dnx[from_node][to_node]:
                            if unfolded_dnx[from_node][to_node]["style"] == "dotted":
                                # unfolded_dnx[from_node][to_node]["label"] = "m"
                                pass
                            else:
                                # unfolded_dnx[from_node][to_node]["label"] = "f>"
                                pass
            pass

        """
        Build control paths using recursive function below.
        Tgen color code the paths if layout is enabled
        """
        control_paths = []

        for node in sorted_nodes:
            super_source_node = "{}_{}".format(source_node, node)

            for to_node, flow in flow_dict[super_source_node].items():
                if flow == 1:
                    control_path = [super_source_node]
                    control_paths.append(control_path)

                    from_node = to_node

                    while True:
                        if control_path[-1] == "t":
                            break

                        if control_path[-1] != from_node:
                            control_path.append(from_node)

                        for to_node, flow in flow_dict[from_node].items():
                            # if flow == 1:
                            if flow > 0:
                                if control_path[-1] == "t":
                                    raise Exception("Can it happent? Double check!")

                                else:
                                    control_path.append(to_node)
                                    from_node = to_node
                                    break
                            pass
                pass
            pass

        if layout:
            distinct_colors = color_set(int(flow_value))
            color_index = 0
            for control_path in control_paths:
                from_node = control_path[0]
                for to_node in control_path[1:]:
                    unfolded_dnx[from_node][to_node]["color"] = distinct_colors[color_index]
                    from_node = to_node
                    pass
                color_index += 1
            pass

        """
        Find all source to sink paths if enabled.
        """
        source_to_sink_paths = []

        if find_all_source_to_sink_paths:
            unfolded_dnx_copy = unfolded_dnx.copy()

            for source_to_sink_path in nx.all_simple_paths(unfolded_dnx_copy, "s", "t"):
                prev_node = None
                is_independent = True
                for node in source_to_sink_path:
                    if prev_node is not None:
                        if flow_dict[prev_node][node] != 1.0:
                            is_independent = False
                            break
                        pass

                    prev_node = node
                    pass
                if is_independent:
                    source_to_sink_paths.append(source_to_sink_path)

            del unfolded_dnx_copy
            pass

        """
        Driver nodes with respect to time
        """

        driver_nodes = dict()
        for node in allowed_drivers:
            driver_nodes[node] = []
            pass

        for node in sorted_nodes:
            super_source_node = "{}_{}".format(source_node, node)

            for stimuli_edge in unfolded_dnx.out_edges(super_source_node):
                # reverse edges
                if stimuli_edge[1] == source_node:
                    continue

                driver_node_id_and_time = stimuli_edge[1].split("_")

                driver_nodes[int(driver_node_id_and_time[1])].append(int(driver_node_id_and_time[0]))

        for node in allowed_drivers:
            if len(driver_nodes[node]) == 0:
                del driver_nodes[node]
            pass

        return unfolded_dnx, flow_value, flow_dict, control_paths, source_to_sink_paths, driver_nodes
        pass
