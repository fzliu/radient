from collections import defaultdict, OrderedDict
from collections.abc import Callable, Iterator
from typing import Any, Dict, List, Optional, Sequence, Union
from graphlib import TopologicalSorter

from radient.utils.flatten_inputs import flattened


class Workflow:
    """Workflows are used to chain together independent tasks together in a
    DAG. The output of each task is maintained in a table and passed to
    subsequent tasks that need the corresponding result.
    """

    def __init__(self):
        self._runners = OrderedDict()
        self._dependencies = {}
        self._runner_graph = None

    def __call__(self, *args, **kwargs) -> Any:
        self.compile()
        return self.execute(*args, **kwargs)

    def add(
        self,
        runner: Callable,
        name: str,
        dependencies: Optional[Sequence[str]] = None
    ) -> "Workflow":

        # By default, new tasks have a single dependency: the preceding task.
        if not dependencies:
            names = list(self._runners.keys())
            dependencies = (names[-1],) if names else ()
        self._dependencies[name] = dependencies

        self._runners[name] = runner

        return self

    def compile(self):
        self._runner_graph = TopologicalSorter(self._dependencies)
        self._all_outputs = defaultdict(list)

    def execute(
        self,
        extra_vars: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        if self._runner_graph is None:
            raise ValueError("call compile() first")

        extra_vars = extra_vars or {}

        # TODO(fzliu): workflows may be persistent rather than returning a
        # single output or set of outputs
        for name in self._runner_graph.static_order():
            inputs = []
            if not self._dependencies[name]:
                # A task with no dependencies is a "seed" task.
                inputs.append([kwargs])
            else:
                for d in self._dependencies[name]:
                    inputs.append(self._all_outputs[d][-1])

            # A task can return a single item or multiple items in a list.
            outputs = []
            for args, _ in flattened(*inputs):
                kwargs = {k: v for d in args for k, v in d.items()}
                kwargs.update(extra_vars.get(name, {}))
                result = self._runners[name](**kwargs)
                if isinstance(result, list):
                    outputs.extend(result)
                else:
                    outputs.append(result)
            self._all_outputs[name].append(outputs)

        return self._all_outputs[name]


