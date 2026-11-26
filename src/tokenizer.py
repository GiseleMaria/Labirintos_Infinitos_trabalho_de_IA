import torch
COND_OFFSET=16
VOCAB_SIZE=26
SEQ_LEN=101
def chunk_to_seq(item: dict) -> list[int]:
    return [COND_OFFSET + item["in_row"]] + item["grid"]

