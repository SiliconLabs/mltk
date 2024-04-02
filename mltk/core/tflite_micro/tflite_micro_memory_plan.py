from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass
import copy
from mltk.utils.string_formatting import format_units
from mltk.core.tflite_model import TfliteModel, TfliteOpCode


MEMORY_ALIGNMENT = 16

def align_memory(value:int) -> int:
    return ((value + MEMORY_ALIGNMENT - 1) // MEMORY_ALIGNMENT) * MEMORY_ALIGNMENT


@dataclass
class TfliteMicroMemoryPlanBufferTensor:
    """This is a helper class containing information about a buffer's
    corresponding tensor
    """
    layer_index:int=-1
    """The tensor's layer index"""
    layer_opcode:TfliteOpCode=None
    """The tensor's layer opcode"""
    layer_name:str=None
    """The name of the tensor's layer"""
    size_bytes:int=0
    """The number of bytes used by the tensor"""
    label:str=None
    """A string label for the tensor. e.g. "input1", "output2" """

    @property
    def name(self) -> str:
        """A generated name for this tensor buffer, e.g." op15-conv2d:input1"""
        if self.layer_name:
            return f'{self.layer_name}:{self.label}'
        return self.label

    def __str__(self):
        return self.name


@dataclass
class TfliteMicroMemoryPlanBuffer:
    """A non-persistent buffer within a memory plan

    Buffers are used by:
    - Model runtime tensors (e.g. activations)
    - Scratch buffers (buffer that are only needed while a particular layer executes)
    """
    size:int
    """The number of bytes required by this buffer"""
    start:int
    """The first time (i.e. layer index) this buffer is used"""
    end:int
    """The last time (i.e. layer index) this buffer is used"""
    offset:int = -1
    """The byte offset from the beginning of the runtime buffer.
    buffer_address = &runtime_buffer[offset]
    """

    tensors:List[TfliteMicroMemoryPlanBufferTensor] = None
    """List of tensors that use this buffer. 
    If this buffer is not used by any tensors (e.g. scratch buffer), then this is None"""
    subgraph_id:int = -1 
    """.tflite flatbuffer index of the tensor's subgraph that uses this buffer"""
    tensor_id:int = -1
    """.tflite flatbuffer index of the tensor that uses this buffer"""

    @property
    def associated_tensors_str(self) -> str:
        """Comma-separated list of tensor names that use this buffer"""
        if not self.tensors:
            return ''
        
        return ','.join(x.name for x in self.tensors)
        
    @property
    def max_tensor(self) -> TfliteMicroMemoryPlanBufferTensor:
        """Return the TfliteMicroMemoryPlanBufferTensor corresponding to the largest tensor that uses this buffer"""
        return None if self.tensors is None else sorted(self.tensors, reverse=True, key=lambda x: x.size_bytes)[0]

    @property
    def memory_usage_bytes(self) -> int:
        """Return the number of bytes that this buffer consumes in the runtime memory.
        The number of bytes = &runtime_buffer[offset + size] - runtime_buffer
        """
        return self.offset + self.size 
        
    def __hash__(self):
        return hash((self.size, self.start, self.end, self.offset))
    
    def equals(
        self, 
        buffer:TfliteMicroMemoryPlanBuffer=None,
        subgraph_id:int=None, 
        tensor_id:int=None
    ) -> bool:
        """Return if the given buffer or subgraph_id/tensor_id matches this buffer's subgraph_id/tensor_id"""
        if buffer:
            if buffer.subgraph_id == -1 or buffer.tensor_id == -1:
                return False 
            return buffer.tensor_id == self.tensor_id and buffer.subgraph_id == self.subgraph_id
        elif subgraph_id is not None and tensor_id is not None:
            return tensor_id == self.tensor_id and subgraph_id == self.subgraph_id
    
        raise ValueError('Must provide buf or subgraph_id and tensor_id args')

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TfliteMicroMemoryPlanBuffer):
            raise ValueError('Source of equality must be a TfliteMicroMemoryPlanBuffer instance')
        return self.equals(buffer=value)
    

class TfliteMicroMemoryPlanBufferGroup(Tuple[TfliteMicroMemoryPlanBuffer]):
    """A tuple of buffers that are all active at the same time in the runtime memory
    
    Typically, these buffers map to the input/output tensors of a given layer.
    """
    def __new__ (cls, time:int, buffers:List[TfliteMicroMemoryPlanBuffer]):
        return super().__new__(cls, tuple(buffers))
    def __init__(self, time:int, buffers:List[TfliteMicroMemoryPlanBuffer]):
        self.time = time

    @property
    def buffer_memory_usage_bytes(self) -> int:
        """The number of bytes this buffer group consumes in the runtime buffer.
        
        Typically, this is the number of bytes consumed in the runtime memory by a given layers input/output tensors.
        """
        max_size = 0
        for buffer in self:
            size = buffer.memory_usage_bytes
            if size > max_size:
                max_size = size
        return align_memory(max_size)


    
class TfliteMicroMemoryPlan(List[TfliteMicroMemoryPlanBuffer]):
    """Specifies how non-persistent buffers are laid out within the runtime memory.
    
    Buffers are used by:
    - Model runtime tensors (e.g. activations)
    - Scratch buffers (buffer that are only needed while a particular layer executes)

    To reduce memory usage, the runtime memory is shared between
    the buffers in the memory plan. When a given buffer is no longer
    needed in the model's execution graph, the buffer's memory can be used
    by other buffers in the graph.

    The memory plan specifies when each buffer is "valid" as the layers of
    the corresponding ML model executes.
    """
    def __init__(
        self, 
        memory_type:str='sram',
        tflite_model:TfliteModel=None,
        total_persistent_runtime_size:int=None,
        persistent_runtime_sizes:List[int]=None,
        temp_runtime_sizes:List[int]=None
    ):
        self.memory_type = memory_type
        """The type of memory used by the buffers of this memory plan. This is just for bookkeeping"""
        self.tflite_model = tflite_model
        """The corresponding .tflite model instance used to generate this memory plan"""
        self.total_persistent_runtime_size = total_persistent_runtime_size or 0
        """The total number of "persistent" bytes used. This is the amount of ram that is always valid
        and not shared with the "non-persistent" buffers of this memory plan.
        NOTE: While this includes the persistent_runtime_sizes (i.e layer persistent allocs), 
        it also includes other global persistent buffer allocs
        """
        self.persistent_runtime_sizes = persistent_runtime_sizes or []
        """The persistent memory used by each model layer"""
        self.temp_runtime_sizes = temp_runtime_sizes or []
        """The amount of temporary memory used by each model layer"""

    def clone(self) -> TfliteMicroMemoryPlan:
        """Create a deep copy of the current plan"""
        plan = TfliteMicroMemoryPlan(
            memory_type=self.memory_type,
            tflite_model=self.tflite_model,
            total_persistent_runtime_size=self.total_persistent_runtime_size,
            persistent_runtime_sizes=self.persistent_runtime_sizes,
            temp_runtime_sizes=self.temp_runtime_sizes
        )
        for buf in self:
            plan.append(copy.deepcopy(buf))

        return plan
    
        
    @staticmethod
    def create(
        memory_plan:List[dict],
        tflite_model:TfliteModel=None,
        memory_type:str='sram',
        total_persistent_runtime_size:int=None,
        persistent_runtime_sizes:List[int]=None,
        temp_runtime_sizes:List[int]=None
    ) -> TfliteMicroMemoryPlan:
        """"Create a memory plan from the recorded results of the TFLM wrapper"""
        plan = TfliteMicroMemoryPlan(
            memory_type=memory_type,
            tflite_model=tflite_model,
            total_persistent_runtime_size=total_persistent_runtime_size,
            persistent_runtime_sizes=persistent_runtime_sizes,
            temp_runtime_sizes=temp_runtime_sizes
        )
        for e in memory_plan:
            buffer = TfliteMicroMemoryPlanBuffer(
                offset = e['offset'],
                start = e['start'],
                end = e['end'],
                size = e['size'],
                subgraph_id=e.get('subgraph_id', -1),
                tensor_id=e.get('tensor_id', -1)
            )

            _get_tflite_tensor(
                tflite_model=tflite_model, 
                buffer=buffer
            )

            plan.append(buffer)

        return plan

    @property
    def buffer_memory_usage_bytes(self) -> int:
        """Return the maximum number of bytes required to store the non-persistent buffers of this memory plan"""
        max_size = 0
        for buffer in self:
            size = buffer.memory_usage_bytes
            if size > max_size:
                max_size = size
        return align_memory(max_size)


    @property 
    def total_memory_usage_bytes(self) -> int:
        """Return the total memory used by the non-persistent buffers, persistent buffers, and temporary allocations"""
        max_layer_size = 0
        temp_runtime_sizes = self.temp_runtime_sizes or [0] * len(self.buffer_groups)
        for buf, temp_memory_used in zip(self.buffer_groups, temp_runtime_sizes + [temp_runtime_sizes[-1]]):
            temp_memory_used_aligned = align_memory(temp_memory_used)
            layer_size = buf.buffer_memory_usage_bytes + temp_memory_used_aligned
            if layer_size > max_layer_size:
                max_layer_size = layer_size
    
        total_size = self.total_persistent_runtime_size + max_layer_size
        if total_size > 0:
            total_size += 256
    
        return total_size

    @property 
    def temporary_memory_usage_bytes(self) -> int:
        """Return the maximum memory required to store the temporary allocations"""
        max_layer_size = 0
        for temp_memory_used in self.temp_runtime_sizes:
            temp_memory_used_aligned = align_memory(temp_memory_used)
            if temp_memory_used_aligned > max_layer_size:
                max_layer_size = temp_memory_used_aligned
    
        return max_layer_size

    @property
    def max_time(self) -> int:
        """Return the maximum time step required by this memory plan. 
        This typically maps to the number of layers in the model"""
        max_time = 0
        for buffer in self:
            if buffer.end > max_time:
                max_time = buffer.end
        return max_time

    @property
    def buffer_groups(self) -> List[TfliteMicroMemoryPlanBufferGroup]:
        """Return a list of tuples containing the groups of non-persistent buffers that are active at the same time.
        
        Typically, each entry in the list maps to each layer of the model.
        And each buffer in an entry maps to the input/output/scratch tensors
        used by a layer.
        """
        retval:List[TfliteMicroMemoryPlanBufferGroup] = []
        for t in range(self.max_time+1):
            buffers = []
            for buffer in self:
                if buffer.offset == -1:
                    continue
                if t < buffer.start or t > buffer.end:
                    continue

                buffers.append(buffer)
    
            retval.append(TfliteMicroMemoryPlanBufferGroup(t, buffers))
    
        return retval
    

    @property
    def n_valid_buffers(self) -> int:
        """Return the number of "valid" non-persistent buffers in this memory plan
        A buffer is "invalid" if its start and end attributes are -1.
        """
        return sum(1 for x in self if x.start >= 0 and x.end >= 0)


    def to_string(self, line_width=80) -> str:
        """Convert the memory plan to a human-readable string"""

        retval = f'{self.memory_type.upper()} Memory Usage:\n'
        retval += self.get_summary() + '\n'

        retval += 'Non-Persistent Buffer Layout:\n'
        retval += self.get_layout_str(line_width=line_width) 

        return retval
    

    def get_summary(self, include_layer_stats=False) -> str:
        """Return a string summary of the memory usage"""
        retval = ''
        retval += f'Total usage: {_format_units(self.total_memory_usage_bytes)} (Total {self.memory_type} memory required)\n'
        retval += f'  Persistent = {_format_units(self.total_persistent_runtime_size)} (Persists for the life of the model)\n'
        retval += f'  Temporary  = {_format_units(self.temporary_memory_usage_bytes)} (Max temporary allocations used by layers)\n'
        retval += f'  Buffers    = {_format_units(self.buffer_memory_usage_bytes)} (Max memory used by layer activation tensor buffers, scratch buffers, etc.)\n'
                
        if include_layer_stats and self.tflite_model and (self.temp_runtime_sizes or self.persistent_runtime_sizes):
            max_layer_name = 0
            for layer in self.tflite_model.layers:
                if len(layer.opcode_str) > max_layer_name:
                    max_layer_name = len(layer.name) + 2

            retval += 'Layer Temporary & Persistent Usage:\n'
            for i, layer in enumerate(self.tflite_model.layers):
                temp_size = self.temp_runtime_sizes[i] if i < len(self.temp_runtime_sizes) else 0
                persistent_size = self.persistent_runtime_sizes[i] if i < len(self.persistent_runtime_sizes) else 0
                retval += f'{layer.name.ljust(max_layer_name)}: temp={_format_units(temp_size)} persist={_format_units(persistent_size)}\n'
            
        return retval.rstrip()

    def get_layout_str(self, line_width=80) -> str:
        """Return a human-readable string layout of the non-persistent memory plan"""
        def _get_ordinal_chr(i:int) -> int:
            if i < 10:
                return ord('0') + i
            elif i < 10 + 26:
                return ord('a') + (i - 10)
            elif i < 10 + 26 + 26:
                return ord('A') + (i - 10 - 26)
            return ord('*')
        
        max_time = self.max_time
        if max_time == 0:
            return ''
        
        max_size = self.buffer_memory_usage_bytes
        max_layer_name = 0
        if self.tflite_model is not None and len(self.tflite_model.layers) == max_time:
            for layer in self.tflite_model.layers:
                if len(layer.opcode_str) > max_layer_name:
                    max_layer_name = len(layer.opcode_str)
            
        retval = ''
        for t in range(max_time+1):
            line = list('.' for _ in range(line_width))
            memory_use = 0

            buffer_id = -1
            for buffer in self:
                if buffer.offset == -1 or buffer.size == 0:
                    continue

                buffer_id += 1

                if t < buffer.start or t > buffer.end:
                    continue

                size = buffer.size
                offset = buffer.offset
                memory_use += size 
                line_start = (offset * line_width) // max_size
                line_end = ((offset + size) * line_width) // max_size
                if line_end > line_width:
                    raise RuntimeError('Invalid line length')
                for n in range(line_start,line_end):
                    if line[n] == '.':
                        line[n] = chr(_get_ordinal_chr(buffer_id))
                    else:
                        line[n] = '!'


            row_label = f'{t:2d}:'
            if max_layer_name and t < len(self.tflite_model.layers):
                layer_opcode = self.tflite_model.layers[t].opcode_str
                row_label += f'{layer_opcode}:'
            row_label = row_label.ljust(4 + max_layer_name)
                
            retval += f'{row_label} {"".join(line)} ({_format_units(memory_use)})\n'

        retval += '\nLegend:\n'
        buffer_id = -1
        for buffer in self:
            if buffer.offset == -1 or buffer.size == 0:
                continue
            buffer_id += 1
            buffer_id_str = str(chr(_get_ordinal_chr(buffer_id)))
            retval += f'{buffer_id_str} -> {buffer.associated_tensors_str}\n'

        retval.rstrip()

        return retval

    def __str__(self) -> str:
        return self.to_string()
    

class TfliteMicroMemoryPlanner:
    """This provides an interface to the Tensorflow-Lite Micro "Greedy Memory Planner":
    https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/memory_planner/greedy_memory_planner.h

    A memory planner that uses a greedy algorithm to arrange buffers in memory
    to minimize the overall arena size needed.
    
    """
    def __init__(
        self, 
        tflite_model:TfliteModel=None,
        memory_type:str='sram'
    ) :
        from .tflite_micro import TfliteMicro

        self.tflite_model = tflite_model
        self.memory_type = memory_type
        wrapper = TfliteMicro._load_wrapper()
        self._planner = wrapper.TfliteMicroMemoryPlannerWrapper()


    def add_buffer(
        self, 
        buffer:TfliteMicroMemoryPlanBuffer
    ):
        """Add a buffer to the memory plan"""
        if not self._planner.add_buffer(
            buffer.size,
            buffer.start,
            buffer.end,
            buffer.offset,
            buffer.subgraph_id,
            buffer.tensor_id,
        ):
            raise RuntimeError('Failed to add buffer')

    def get_plan(self) -> TfliteMicroMemoryPlan:
        """Use the TFLM "Greedy Memory Planner" to generate a memory for the given list of buffers"""
        buffer_list = self._planner.get_plan()
        return TfliteMicroMemoryPlan.create(
            buffer_list, 
            tflite_model=self.tflite_model,
            memory_type=self.memory_type
        )


def _get_tflite_tensor(
    tflite_model:TfliteModel,
    buffer:TfliteMicroMemoryPlanBuffer
):
    """Find all the tensors in the given tflite model instance that
    reference the given buffer"""
    if tflite_model is None:
        return
    
    if buffer.subgraph_id == -1 or buffer.tensor_id == -1:
        if buffer.tensors is None:
            buffer.tensors = [TfliteMicroMemoryPlanBufferTensor(
                layer_name=None,
                size_bytes=None,
                label='scratch/variable'
            )]
        return

    saved_subgraph = tflite_model.selected_model_subgraph
    if buffer.tensors is None:
        buffer.tensors = []

    try:
        tflite_model.selected_model_subgraph = buffer.subgraph_id
        for layer in tflite_model.layers:
            for i, inp in enumerate(layer.inputs):
                if inp and inp.index == buffer.tensor_id:
                    buffer.tensors.append(TfliteMicroMemoryPlanBufferTensor(
                        layer_name=layer.name,
                        layer_index=layer.index,
                        layer_opcode=layer.opcode,
                        size_bytes=inp.size_bytes,
                        label=f'input{i}'
                    ))

            for i, outp in enumerate(layer.outputs):
                if outp.index == buffer.tensor_id:
                    buffer.tensors.append(TfliteMicroMemoryPlanBufferTensor(
                        layer_name=layer.name,
                        layer_index=layer.index,
                        layer_opcode=layer.opcode,
                        size_bytes=outp.size_bytes,
                        label=f'output{i}'
                    ))

    finally:
        tflite_model.selected_model_subgraph = saved_subgraph



def _format_units(v:int) -> str:
    return format_units(v, precision=1, add_space=False, rjust=7)
