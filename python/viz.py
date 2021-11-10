#!/usr/bin/python3
#
# Copyright (C) 2021 Richard Preen <rpreen@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Class for generating graphviz plots of GP trees.
"""

import graphviz

class TreeViz:
    """ Visualises a GP tree with graphviz. """

    def __init__(self, tree, filename, feature_names=None):
        """ Plots a tree with graphviz, saving to filename.
        Optionally uses a list of feature names. """
        self.feature_names = feature_names
        self.tree = tree
        self.cnt = 0
        self.pos = 0
        self.gviz = graphviz.Digraph('G', filename=filename+'.gv')
        self.read_subexpr()
        self.gviz.view()

    def label(self, symbol):
        """ Returns the node label for a symbol. """
        if not self.feature_names is None and isinstance(symbol, str):
            start, end = symbol.split('_') if '_' in symbol else (symbol, '')
            if start == 'feature' and int(end) < len(self.feature_names):
                return self.feature_names[int(end)]
        return str(symbol)

    def read_function(self):
        """ Parses functions. """
        expr1 = self.read_subexpr()
        symbol = self.tree[self.pos]
        if symbol in ('+', '-', '*', '/'):
            self.pos += 1
            expr2 = self.read_function()
            self.cnt += 1
            self.gviz.edge(str(self.cnt), str(expr1))
            self.gviz.edge(str(self.cnt), str(expr2))
            self.gviz.node(str(self.cnt), label=self.label(symbol))
            return expr2
        return expr1

    def read_subexpr(self):
        """ Parses sub-expressions. """
        symbol = self.tree[self.pos]
        self.pos += 1
        if symbol == '(':
            self.read_function()
            self.pos += 1 # ')'
        else:
            self.cnt += 1
            self.gviz.node(str(self.cnt), label=self.label(symbol))
        return self.cnt
