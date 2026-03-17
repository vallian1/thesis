import base64
import hashlib
import json
import os
import zlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_B64 = {
    "clean1": (
        "eNollUsW5SAIBTeUgQIirqVP738bXfV6kBujgHwu5E/kl/VVfmd9Z76O7/Y363v72+t8e/Pe8+3gHe/bxbvq26e/3ZeHvRs8rC/r"
        "4Xz4Hr6Hb+28+8XiwVZs3twZ2AouDuxFBU9/gQtxOGvezffljZ149eXmwYfMyzNf4mvt/ip41/kK3ZomivMdfDtEcbi/1/4anSbC"
        "xtbF/sXvQW645534HrJ7bYJdSO6FO3tha4OAextX9n5ALIKKIAORro55aQ8mzRCQqqX2MvxM7CXxAWikwtme3i14MGq8Nr3mV90K"
        "7i3vKHwE/DTTNSVwYLD7qHGCg+NFh0RQHEyd8fRZKy33RqN13Hzs1o3Wjb7uWadroa8i1xxcfb7tgdWdtSxvW1/U5iA3LfzqbeTP"
        "5DzKCiD3yj0df9725Moz1PcT/tFCTpj7WEHFF9QCBmgPZMC67hFCbFwDXB3pBAfgE4TZDwbFllb/+YXuj1nRiMS4oniArNs/6i2B"
        "04Qtke0eNQ8LACFd/WgpH6sF2B318PRsearIgYlxqGocOXvonjj3Bx6ocUhsyMJoo7QK0VQL8DOluu42CQNQa12Tr4DCdkX/2sJs"
        "9FVtNDAKG9ZdIZCcuwX6JCxjXA1cA7xUOq6ZvBAuRo2hWuSGhI0ak+7RSzHeNqrNuDIvb6WNyG1Ph55pf7iRVjBXHgCfc3GQFFN4"
        "tiz9arOnXZ5ECFCtTIiUSfgAp8m9aWsAdHoqYhtkpQOKQtFBaNR1z4FwdgjoHjUOzZR2PEbQMNnZRIQ5V/AgTRiFBm7h1YXAealW"
        "yuwc7Rk0zYjIM4RnbC9cGaDhA9whnwHldOiNgFe1MAUwlBY5BRhNkrrW+YEH1wOqUGtcjZNMtb3Y2zheTBkgBXLKnHM1rgiwggAr"
        "NB/MiIK3wPHzeNq/qYiVJIRKQigpXyQfIJOV1Ld+E5ScAOMP4LlHf1Rp/jdbS++LaVFlCLZBld47fZi4iDh96pRw/NSes7fay5uB"
        "Uu2Ibn8yfdxTrnWyKXJdc3W1cnXy6uSl5iV3S9rWhfd15wcIy90aNUbhUXh0cn7/AGMb2rQe8wBA5OmBxQNcnff3H9hgXxY="
    ),
    "mild": (
        "eNodkcmRxUAIQxPSAcTWxPJr8k9jhA+8YlEDxj+vBmlgEpyRLbiBcEcoH0ykyeIhiyhp6hFtjlZ9bDARssZ0YibwgniVWCvZw7p8"
        "b+wruLEPI2zCRcEP6ub0DwpJCuroHIeHenhcLurDhX2eFvKkOmfobZbE2SGsvKK8ClUr7XBhXvjUoO9F6wLeT954HJ5wunlaY02F"
        "Tc1YSWiRhwe6BmnRD3fCK/D5QVXuhVtg6HhMfRHzJHmFMr2oCmHk3S3ZGs4u6W4XQf+iTzx5GDV9+g6uSbeX21Ru57z5vEWYpoUx"
        "//4BxbVc9Q=="
    ),
    "moderate": (
        "eNodksd1ACEMBRvSYRVQqMXP/bfh+T5oEKAMP5mWzyptP9swf2nhHxIWcxbnSFsW0mXVjpzVJMK6rNtWN/bckbNHjPfYN8L9O0kh"
        "jax1pDV5m1h9gZzN92zIOxU2nM+0DXcbz7bXlv35s+uwG1Zyn+r9XgkDMAMHljNUgS0UpIW0FOjYXQGcLK6WPUJaYZeJbz622djl"
        "SCOrlxO5woEuNAVAcs3A30eUV/9I4QnEe7J7UwJF9sO4mYzPh+8kZyOPoSsfXewn0DDAZNXlMheAyarBZZx+sruPAKccNzpTW7c8"
        "3scM4qNcsCB51K+0pRnALYPRY4eAiVeB1icgfBBBwDjkG6mzkSbfWHzjuE2ZJG8XSaWhqYEBR4Bi9lEhTbWUslW0oAsFLflWafuE"
        "lh1vHZopaIH6HtOIlzrj10Qzv2gFbX5xtEpr5hzjGM+CVbzldwP9Z9mdEt3oa5///gFgjpfm"
    ),
    "burst": (
        "eNollcuVACEIBBPyoICosezb/NPYqtnD9KgI8mn0J3Jkjcqx59h3dIzT487x1lhzj7X4rztW8I83VvGvGmv3WH34WDvBx/gwvsgv"
        "88v8MtfOOyMmH7Zi8efMwFZwcGAvKvh6BC7EYq/5NvPNFztx68vJDx8yN7/zJb7W7K+Cb62v0K3TRLG+hW+LKBbn95hfo9NE2Nja"
        "2N/4fZA7nHNXfBfZOSbBDiTnwJ05sDVBwLWJK3NeIAZBRZCBSEfLvLQbJ80QkKql9jKcJvaS+AA0UuFsd/cU3Dhq3Da95lfdCs4t"
        "zygCAJxaqrolIDBLa6uxA8H2oE0mqS6m9lX6LLaWm1Ss1nGTuFo3Wjf6uGahj0EftxyDPvp8WoH0uHPKj5YgqN3NvtvCRxgjf/MD"
        "eQM5ADa/ck3vn0c+GfdM9jPo96l9DJNeViZmQJ5JhYELbEetVEbN45SIYuEp4GhLT/gAP1OAneuFIF8lbyxZ+09fTH3E/ZgajVpc"
        "p8QRljE+2sZT+iS6urkcQRp4PwXUEopGtgIiCosX0pqWEL7m+LrDtqgWdK2oDICBeoz2snHU2LRGbFgSW9c27Rz7fKBA3a2n29M2"
        "JQsbI9qsWd9oeAA4TbvQUJsqABhoHbd9ADfbsP11rInto5o5aF1rwz8zBPJ8lkALhwSJo4Fj+AcOxbEoByrHVeNCAVJKEq8a8oW8"
        "KsgUMH/LtXKLHlxNXYt8Dfpq9FqKa6jyCmDtae9p5RnMM4RnzR+OpxxKGzsnUeZEkNBJeN4/XD7eXOmVleQEgCWZlCyThAFIE9fS"
        "NgW4ttIttmRWettSeLoZjTquebvtFQK6W41NY6fXVm51G+/TGmWnU803rMumrdJkQyG3PEaHlACOiA3A3oECAPbsQYDDr0degzF1"
        "XCBoXNiUz3hNWD6z8cKpKTFhABpPjUe7pC0JONWhZ0TvCjr+iK0mRwLc05PcA9zQNmfZl4CCo4Dq17yOrpe7agtPaQo0uDeBFKgM"
        "PeLoOiJNFbhbofng1iu6CdhOt9L+HgqsJLFVElbZjUUJAXJVCYfqe1TIJ3B9E59r9G+V5r/npvS+uP+qDMGWrNJ771MeIbZ4n9Yu"
        "YTvVns9QtYc3V2S1r1bn91y55r7WySZrdczV0crRyaOTB+aUPVO2Sx36rc79gM32TPl01XWzTQLg5PVZtD/K0tbjHgLY8vTAggKO"
        "9vv9A1MllY8="
    ),
    "block": (
        "eNodlkuWBCEIBC9UCwVEPUu/uf81JsJFZ/kBBEywf5Ff1lf5rfGt83V8u78zvju/OdY3J995vhl8436z+FZ9c/U3e/NjbQc/xpvx"
        "Yf8wP8wPc+3c/cXgh62YfDkzsBUcHNiLCn79BS7EYs/5NvPNFztx68vJDx8yN7/zJb7W7K+Cb62v0K3TRLG+hW+LKBbn95hfo9NE"
        "2Nja2N/4fZA7nHNXfBfZOSbBDiTnwJ05sDVBwLWJK3NeIAZBRZCBSEfLvLQbJ80QkKql9jKcJvaS+AA0UuFsd/cU3Dhq3Da95lfd"
        "Cs4tzyjVDBAIwQ3in2Xi65SAyBopLAFTS1NrvpG7EYIioYiXuXRt6drS/GpHx93rFetQq9vGaxpn1xDe6AjuLsGwuqcQwhKU2+ru"
        "Rxc1JEnLEq8DcCpJ2iP7SiP5tj1yexVb/7aWtyQ7Y8iylmaccRZyp4VHOy/gekcXdgHI3XLNrF2dvLp25e017fdpPIqiFkOSjhnC"
        "AsJpQMYB64EDyNMhUcdyKl1HK9dqtGtSd2w1trrbtevocsYkNoApNBFSsEDS3RKWU82TfKtGEU3N61R3X0mhIryiwtQrp9CDOI5I"
        "SUjMeHUUEBiw8kgi4EgDSToBp3qVFmbmErZlqrAFmoafUB5wd7m2VDMH2aq1Uz1NSijkM/Xt6FV7Ono1/4reJFYL0CIK6gGcUdfR"
        "ZbSmTUEryyiXpla6YZqW5y4ztPYDNzS1DH/pwYJcYRsAMNrG23QT4E0RbkoDcJq2HjPZetoeJOXD/gG4BvlD8ofkj5YM/bqWd95b"
        "K6a9jaPN+B4hcOT23C2lpHdsDWyTuGXOli/b1G3j2Maxqe6Q8sAUQmDjmI2jqZOu0QPj6MbR3pE+Z7vr7Z9nxWwcc3qkwNXopWXQ"
        "Xh0Zx5VmF+/TMsgBBXIQOUDblfIp0XMglxM3kn5IX8YrwA5N40mJmfbJlJhpq097PJCCa+0IKqfUyyTtAHJJHCmbUjaxhIjtMCt9"
        "qEg7RESjtms+DItaTTscbHRK0GnnxwgaXnI2GcKcI0ojvRloCmx92XSQ3NAnzXMe7ZlEmjIi19iu2bjhyJSYMIAzbCiAcjp0j4BX"
        "NTAF8DjZQQCeKDtIjfXAje0G113jODq+aKpNkkjdoGF2i/cEIHW8d46OIwKsIMAKzQdvRVFwwHK63O33OmIlCaGSEMoSL0u8rORK"
        "LrTeS0pOgOMfgesaBKnS/HtjS++LVlBlCBZsld777JSPTVmhtUpYTrXnG1zt4U1Hr/apbv9s9HJNudbJ5pJrm6utla2TWyc3d14W"
        "SRkftSmw2ucBwtZCHTWOwkfho5Pn/RcwtkMbqUvFA4hcPfDyAEfr/v0DTbnFIA=="
    ),
    "severe": (
        "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"
        "BC1AaOMASnKv79x+fDzVc"
    ),
}

EXPECTED_MD5 = {
    "exp4_real_day1_reference.txt": "5578022b6037e2aaee4eb74fe4d0bf37",
    "exp4_real_day2_reference.txt": "7edb91b649a74ef28ddc6f921aec9fcc",
    "exp4_day1_cleaned.txt": "926efeb4b440e4a013a81a682541ed5e",
    "exp4_case_mild_missing.txt": "22b2417a94a7d8e227101cf2fc24b371",
    "exp4_case_moderate_missing.txt": "df31f90cc55a046d4a0221459175bfb5",
    "exp4_case_burst_missing.txt": "744476ed99d5c21127d6a81d022debed",
    "exp4_case_block_missing.txt": "c903c5bf5e039b42fcd2edb6cdb8e4e2",
    "exp4_case_severe_hybrid.txt": "baddf1c76a4b2b7a017756e59c7aa34e",
}

SOURCES = {
    "day1": os.path.join(SCRIPT_DIR, "streamIn1215.txt"),
    "day2": os.path.join(SCRIPT_DIR, "streamIn1216.txt"),
}

OUTPUTS = {
    "exp4_real_day1_reference.txt": ("day1", None),
    "exp4_real_day2_reference.txt": ("day2", None),
    "exp4_day1_cleaned.txt": ("day1", "clean1"),
    "exp4_case_mild_missing.txt": ("day2", "mild"),
    "exp4_case_moderate_missing.txt": ("day2", "moderate"),
    "exp4_case_burst_missing.txt": ("day1", "burst"),
    "exp4_case_block_missing.txt": ("day1", "block"),
    "exp4_case_severe_hybrid.txt": ("day1", "severe"),
}

def decode_indices(name: str):
    raw = zlib.decompress(base64.b64decode(INDEX_B64[name].encode("ascii")))
    return json.loads(raw.decode("utf-8"))

def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def write_lines(path: str, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if lines:
            f.write("\n".join(lines) + "\n")

def write_reference(src_path: str, dst_path: str):
    lines = read_lines(src_path)
    write_lines(dst_path, lines)
    return len(lines)

def write_subset(src_path: str, dst_path: str, remove_idx):
    lines = read_lines(src_path)
    remove_set = set(remove_idx)
    out_lines = [line for i, line in enumerate(lines) if i not in remove_set]
    write_lines(dst_path, out_lines)
    return len(lines), len(out_lines), len(remove_set)

def main():
    for src_name in SOURCES.values():
        if not os.path.exists(src_name):
            raise FileNotFoundError(f"Missing source file: {src_name}")

    for out_name, (src_key, subset_name) in OUTPUTS.items():
        src_name = SOURCES[src_key]
        out_path = os.path.join(SCRIPT_DIR, out_name)
        if subset_name is None:
            out_rows = write_reference(src_name, out_path)
            removed = 0
        else:
            _, out_rows, removed = write_subset(
                src_name, out_path, decode_indices(subset_name)
            )

        md5_value = file_md5(out_name)
        ok = "OK" if md5_value == EXPECTED_MD5[out_name] else "CHECK"
        print(f"{out_name} | rows={out_rows} | removed={removed} | md5={md5_value} | {ok}")

if __name__ == "__main__":
    main()
