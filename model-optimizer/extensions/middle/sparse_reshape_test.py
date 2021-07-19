"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import unittest

from extensions.middle.sparse_reshape import SparseReshapeMiddleReplacer
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph


class SparseReshapeMiddleReplacerTests(unittest.TestCase):
    def test1(self):
        graph = build_graph({
                             'const_dense_shape': {'type': 'Const', 'kind': 'op', 'op': 'Const',
                                                   'value': int64_array([4, 5]), 'shape': int64_array([2])},
                             'const_dense_shape_data': {'kind': 'data',
                                                        'value': int64_array([4, 5]), 'shape': int64_array([2])},
                             'const_new_dense_shape': {'type': 'Const', 'kind': 'op', 'op': 'Const',
                                                   'value': int64_array([4, -1]), 'shape': int64_array([2])},
                             'const_new_dense_shape_data': {'kind': 'data',
                                                        'value': int64_array([4, -1]), 'shape': int64_array([2])},
                             'const_default_value': {'type': 'Const', 'kind': 'op', 'op': 'Const',
                                                   'value': 2, 'shape': int64_array([])},
                             'const_default_value_data': {'kind': 'data','value': 2, 'shape': int64_array([])},

                             'input_indices': {'value': None, 'shape': int64_array([10, 2]), 'type': 'Parameter', 'kind': 'op',
                                               'op': 'Parameter'},
                             'input_indices_data': {'shape': int64_array([10, 2]), 'value': None, 'kind': 'data'},
                             'input_values': {'value': None, 'shape': int64_array([10]), 'type': 'Parameter', 'kind': 'op',
                                              'op': 'Parameter'},
                             'input_values_data': {'shape': int64_array([10]), 'value': None, 'kind': 'data'},
                             'input_params_table': {'value': None, 'shape': int64_array([100, 4, 3]), 'type': 'Parameter', 'kind': 'op',
                                                    'op': 'Parameter'},
                             'input_params_table_data': {'shape': int64_array([10, 4, 3]), 'value': None, 'kind': 'data'},

                             'sparse_reshape': {'kind': 'op', 'op': 'SparseReshape'},

                             'output_indices_data': {'shape': int64_array([10, 2]), 'value': None, 'kind': 'data'},
                             'output_new_dense_shape_data': {'kind': 'data', 'value': int64_array([4, 5]), 'shape': int64_array([2])},

                             'sparse_weighted_sum': {'kind': 'op', 'op': 'SparseWeightedSum'},
                             },
                            [
                             ('input_indices', 'input_indices_data'),
                             ('input_indices_data', 'sparse_reshape', {'in': 0}),
                             ('const_dense_shape', 'const_dense_shape_data'),
                             ('const_dense_shape_data', 'sparse_reshape', {'in': 1}),
                             ('const_new_dense_shape', 'const_new_dense_shape_data'),
                             ('const_new_dense_shape_data', 'sparse_reshape', {'in': 2}),
                             ('sparse_reshape', 'output_indices_data', {'out': 0, 'in': 0}),
                             ('sparse_reshape', 'output_new_dense_shape_data', {'out': 1, 'in': 0}),
                             ('output_indices_data', 'sparse_weighted_sum', {'in': 0}),
                             ('input_values', 'input_values_data'),
                             ('input_values_data', 'sparse_weighted_sum', {'in': 1}),
                             ('output_new_dense_shape_data', 'sparse_weighted_sum', {'in': 2}),
                             ('input_params_table', 'input_params_table_data'),
                             ('input_params_table_data', 'sparse_weighted_sum', {'in': 3}),
                             ('const_default_value', 'const_default_value_data'),
                             ('const_default_value_data', 'sparse_weighted_sum', {'in': 4})
                             ])
        SparseReshapeMiddleReplacer().find_and_replace_pattern(graph)
        #graph_clean_up(graph)
        ref_graph = build_graph({
                                 'const_dense_shape': {'type': 'Const', 'kind': 'op', 'op': 'Const',
                                                       'value': int64_array([4, 5]), 'shape': int64_array([2])},
                                 'output_new_dense_shape_data': {'kind': 'data',
                                                            'value': int64_array([4, 5]), 'shape': int64_array([2])},
                                 'const_default_value': {'type': 'Const', 'kind': 'op', 'op': 'Const',
                                                       'value': 2, 'shape': int64_array([])},
                                 'const_default_value_data': {'kind': 'data','value': 2, 'shape': int64_array([])},
                                 'input_indices': {'value': None, 'shape': int64_array([10, 2]), 'type': 'Parameter', 'kind': 'op',
                                                   'op': 'Parameter'},
                                 'output_indices_data': {'shape': int64_array([10, 2]), 'value': None, 'kind': 'data'},
                                 'input_values': {'value': None, 'shape': int64_array([10]), 'type': 'Parameter', 'kind': 'op',
                                                  'op': 'Parameter'},
                                 'input_values_data': {'shape': int64_array([10]), 'value': None, 'kind': 'data'},
                                 'input_params_table': {'value': None, 'shape': int64_array([100, 4, 3]), 'type': 'Parameter', 'kind': 'op',
                                                        'op': 'Parameter'},
                                 'input_params_table_data': {'shape': int64_array([10, 4, 3]), 'value': None, 'kind': 'data'},
                                 'sparse_weighted_sum': {'kind': 'op', 'op': 'SparseWeightedSum'},
                                },
                                [
                                 ('input_indices', 'output_indices_data'),
                                 ('output_indices_data', 'sparse_weighted_sum', {'in': 0}),
                                 ('input_values', 'input_values_data'),
                                 ('input_values_data', 'sparse_weighted_sum', {'in': 1}),
                                 ('const_dense_shape', 'output_new_dense_shape_data'),
                                 ('output_new_dense_shape_data', 'sparse_weighted_sum', {'in': 2}),
                                 ('input_params_table', 'input_params_table_data'),
                                 ('input_params_table_data', 'sparse_weighted_sum', {'in': 3}),
                                 ('const_default_value', 'const_default_value_data'),
                                 ('const_default_value_data', 'sparse_weighted_sum', {'in': 4})
                             ])

        (flag, resp) = compare_graphs(graph, ref_graph, 'sparse_weighted_sum')
        self.assertTrue(flag, resp)