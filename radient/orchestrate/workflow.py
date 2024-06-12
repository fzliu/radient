from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any, Dict, List, Optional, Sequence, Union
from graphlib import TopologicalSorter


class Workflow:
    """Workflows are used to chain together independent Tasks together in a
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
        dependencies: Optional[Sequence[str]] = None
    ) -> "Workflow":

        # By default, new tasks have a single dependency: the preceding task.
        if not dependencies:
            names = list(self._runners.keys())
            dependencies = (names[-1],) if names else ()
        self._dependencies[runner.name] = dependencies

        self._runners[runner.name] = runner

        return self

    def compile(self):
        self._runner_graph = TopologicalSorter(self._dependencies)
        self._outputs = {}

    def execute(self, data: Any) -> Any:
        if self._runner_graph is None:
            raise ValueError("call flow.compile() first")

        for name in self._runner_graph.static_order():
            if not self._dependencies[name]:
                # A task with no dependencies is a "seed" task.
                inputs = (data,)
            else:
                inputs = tuple(self._outputs[d] for d in self._dependencies[name])
            self._outputs[name] = self._runners[name](*inputs)

        return self._outputs[name]


