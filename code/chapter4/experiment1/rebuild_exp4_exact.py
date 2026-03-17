import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..','data')

def read_lines(path):
    full_path = os.path.join(DATA_DIR, path) if not os.path.isabs(path) else path
    with open(full_path,'r',encoding='utf-8') as f:
        return f.read().splitlines()

def write_lines(path,lines):
    full_path = os.path.join(SCRIPT_DIR, path) if not os.path.isabs(path) else path
    with open(full_path,'w',encoding='utf-8') as f:
        for l in lines:
            f.write(l+'\n')

def remove_by_index(lines, idx):
    idx=set(idx)
    return [l for i,l in enumerate(lines) if i not in idx]


# ===== 删除位置 =====

CLEAN_REMOVE = [
23,34,43,50,58,62,76,80,91,105,111,118,121,129,141,144,
156,167,169,172,177,179,181,182,187,189,191,197,207
]

MILD_REMOVE = [
156,220,242,277,279,293,311,320,324,404
]

MOD_REMOVE = [
33,35,43,80,82,153,210,212,279
]

BURST_REMOVE = [
411,412,576,583,597
]

BLOCK_REMOVE = [
1350,1351,1357
]

SEVERE_REMOVE = [
8,13,19,27,35
]


def main():

    day1 = read_lines("streamIn1215.txt")
    day2 = read_lines("streamIn1216.txt")

    write_lines("exp4_real_day1_reference.txt", day1)
    write_lines("exp4_real_day2_reference.txt", day2)

    cleaned = remove_by_index(day1, CLEAN_REMOVE)
    write_lines("exp4_day1_cleaned.txt", cleaned)

    mild = remove_by_index(day2, MILD_REMOVE)
    write_lines("exp4_case_mild_missing.txt", mild)

    mod = remove_by_index(day2, MOD_REMOVE)
    write_lines("exp4_case_moderate_missing.txt", mod)

    burst = remove_by_index(cleaned, BURST_REMOVE)
    write_lines("exp4_case_burst_missing.txt", burst)

    block = remove_by_index(cleaned, BLOCK_REMOVE)
    write_lines("exp4_case_block_missing.txt", block)

    severe = remove_by_index(cleaned, SEVERE_REMOVE)
    write_lines("exp4_case_severe_hybrid.txt", severe)

    print("datasets generated")

if __name__ == "__main__":
    main()