from collections import defaultdict, OrderedDict
from collections.abc import Callable, Iterator
from typing import Any, Dict, List, Optional, Sequence, Union
from graphlib import TopologicalSorter

from radient.runners import LazyLocalRunner
from radient.sinks import make_sink
from radient.transforms import make_transform
from radient.vectorizers import make_vectorizer


def traverse(
    data: Union[Any, List[Any], Dict[str, Union[Any, List[Any]]]]
) -> Iterator:
    """Traverse a nested data structure and yield its elements.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            yield from traverse(value)
    elif isinstance(data, list):
        for item in data:
            yield from traverse(item)
    else:
        yield data


class Flow:
    """Flows are used to chain together independent Tasks together in a DAG.
    The output of each task is maintained in a table and passed to subsequent
    tasks that need the corresponding result.
    """

    def __init__(self):
        self._task_data = OrderedDict()
        self._task_graph = None

    def __call__(self, *args, **kwargs) -> Any:
        self.compile()
        return self.execute(*args, **kwargs)

    def _executor(self, name: str) -> Callable:
        return self._task_data[name][0]

    def add(
        self,
        task: str,
        runner: Callable = LazyLocalRunner,
        flatten_inputs: bool = False,
        name: Optional[str] = None,
        dependencies: Optional[Sequence[str]] = None,
        task_kwargs: Optional[Dict[str, Any]] = None
    ) -> "Flow":

        # Some examples of what the task string should look like:
        # - "source.gdrive"
        # - "transform.video_demux"
        # - "vectorizer.text.sbert"
        # - "sink.milvus"
        parts = task.split(".")
        task_type = parts[0]
        task_args = parts[1:]
        task_kwargs = task_kwargs or {}
        if task_type == "source":
            instance = runner(make_source, *task_args, **task_kwargs)
        elif task_type == "transform":
            instance = runner(make_transform, *task_args, **task_kwargs)
        elif task_type == "vectorizer":
            instance = runner(make_vectorizer, *task_args, **task_kwargs)
        elif task_type == "sink":
            instance = runner(make_sink, *task_args, **task_kwargs)

        # Each task must have a unique name; if no name is provided, we
        # generate one that looks like `_taskN`, where N is the index in which
        # the task was added.
        if not name:
            name = f"_task{len(self._task_data)}"
        if name in self._task_data:
            raise ValueError(f"task {name} already defined")

        # By default, new tasks have a single dependency: the preceding task.
        if not dependencies:
            names = list(self._task_data.keys())
            dependencies = (names[-1],) if names else ()

        # Each task is associated with three pieces of data:
        # 1) An instance of a runner/executor,
        # 2) A sequence of dependencies,
        # 3) A bool indicating whether to flatten the inputs.
        self._task_data[name] = (instance, dependencies, flatten_inputs)

        return self

    def compile(self):
        dependencies = {s: self._task_data[s][1] for s in self._task_data}
        self._task_graph = TopologicalSorter(dependencies)
        self._outputs = defaultdict(None)

    def execute(self, data: Any) -> Any:
        if self._task_graph is None:
            raise ValueError("call flow.compile() first")

        for name in self._task_graph.static_order():
            if not self._task_data[name][1]:
                # A task with no dependencies is a "seed" task.
                inputs = (data,)
            else:
                inputs = [self._outputs[d] for d in self._task_data[name][1]]

            # Maintain data that is output by each task in a table.
            if self._task_data[name][2]:
                # TODO(fzliu): outputs needs to match input dtype (e.g. dict)
                outputs = [self._task_data[name][0](d) for d in traverse(inputs)]
                self._outputs[name] = outputs
            else:
                self._outputs[name] = self._task_data[name][0](*inputs)

        return self._outputs[name]


