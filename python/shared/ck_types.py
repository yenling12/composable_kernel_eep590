# The structure for defining types is taken from Meta's AIT library

from dataclasses import dataclass

class DataType:
    f16 = "ck::half_t"

class Layout:
    ColumnMajor = "ck::tensor_layout::gemm::ColumnMajor"
    RowMajor = "ck::tensor_layout::gemm::RowMajor"

class TensorOperation:
    PassThrough = "ck::tensor_operation::element_wise::PassThrough"

@dataclass
class TensorDesc:
    element: DataType
    layout: Layout

