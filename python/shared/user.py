# The structure for creating a list of instances for an op 
# was taken from Meta's AIT library 

import gemm_op as gemm
import enum
from dataclasses import dataclass
from enum import auto
import ck_types
from ck_types import *

def CreateGemmOperator():
    a_element_desc = TensorDesc(
       DataType.f16, Layout.ColumnMajor
    )
    b_element_desc = TensorDesc(
        DataType.f16, Layout.RowMajor
    )
    c_element_desc = TensorDesc(
        DataType.f16,Layout.RowMajor
    )
    element_op = TensorOperation.PassThrough

    tile_descriptions = [
        gemm.TileDesc(256, 128, 128, 16, 2, 4, 4, 1, "S<8, 2>", "S<8, 2>"),
        gemm.TileDesc(256, 128, 128, 8, 2, 4, 4, 1, "S<8, 2>", "S<8, 2>"),
        gemm.TileDesc(128, 64, 128, 8, 2, 4, 4, 1, "S<4, 2>", "S<8, 2>"),
        gemm.TileDesc(128, 128, 64, 8, 2, 4, 4, 1, "S<8, 2>", "S<4, 2>"),
        gemm.TileDesc(256, 64, 128, 8, 2, 2, 4, 1, "S<8, 2>", "S<8, 2>"),
        gemm.TileDesc(256, 128, 64, 8, 2, 4, 2, 1, "S<8, 2>", "S<8, 2>"),
        gemm.TileDesc(128, 32, 128, 8, 2, 2, 4, 1, "S<4, 2>", "S<8, 2>"),
        gemm.TileDesc(128, 128, 32, 8, 2, 4, 2, 1, "S<8, 2>", "S<4, 2>"),
        gemm.TileDesc(128, 32, 64, 8, 2, 2, 2, 1, "S<4, 2>", "S<8, 2>"),
        gemm.TileDesc(128, 64, 32, 8, 2, 2, 2, 1, "S<8, 2>", "S<4, 2>"),
    ]

    a_block_descriptions = [
        gemm.BlockTransferDesc("S<2, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 8, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 8, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        
    ]

    b_block_descriptions = [
        gemm.BlockTransferDesc("S<2, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 8, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 8, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
    ]

    c_block_descriptions = [
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 2),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 2),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 2),
    ]

    gemm_specialization = [
        gemm.GemmType.GemmDefault
    ]
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, a_block_desc, b_block_desc, c_block_desc in zip(
            tile_descriptions,
            a_block_descriptions,
            b_block_descriptions,
            c_block_descriptions,
        ):
            new_operation = gemm.GemmOperation(
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=a_block_desc,
                b_block_transfer=b_block_desc,
                c_block_transfer=c_block_desc,
            )
            operations.append(new_operation)
    return operations

