import os
import json
import time
from pathlib import Path

import torch
import torchvision
import ezkl


# ------------------------
# Paths
# ------------------------
OUT_DIR = Path("ezkl_resnet18")
OUT_DIR.mkdir(exist_ok=True)

ONNX_PATH = str(OUT_DIR / "resnet18.onnx")
SETTINGS_PATH = str(OUT_DIR / "settings.json")
COMPILED_CIRCUIT = str(OUT_DIR / "resnet18.ezkl")
SRS_PATH = str(OUT_DIR / "kzg.srs")
VK_PATH = str(OUT_DIR / "vk.key")
PK_PATH = str(OUT_DIR / "pk.key")
WITNESS_PATH = str(OUT_DIR / "witness.json")
PROOF_PATH = str(OUT_DIR / "proof.json")
INPUT_JSON = str(OUT_DIR / "input.json")


# ------------------------
# Helpers
# ------------------------
def timed(label, fn, *args, **kwargs):
    t0 = time.time()
    res = fn(*args, **kwargs)
    t1 = time.time()
    print(f"[TIMING] {label}: {t1 - t0:.2f} sec")
    return res


# ------------------------
# 1) Export ResNet18 to ONNX
# ------------------------
def export_resnet18():
    if os.path.exists(ONNX_PATH):
        print(f"[i] Found existing {ONNX_PATH}, skipping export.")
        return
    model = torchvision.models.resnet18(weights="DEFAULT").eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"[+] Exported ResNet18 to {ONNX_PATH}")

    # Save dummy input for witness generation later
    with open(INPUT_JSON, "w") as f:
        json.dump({"input": dummy.numpy().tolist()}, f)


# ------------------------
# 2) Generate + calibrate settings
# ------------------------
def generate_settings():
    if not os.path.exists(SETTINGS_PATH):
        ezkl.gen_settings(ONNX_PATH)
        if os.path.exists("settings.json"):
            os.replace("settings.json", SETTINGS_PATH)
    print(f"[+] Generated settings at {SETTINGS_PATH}")

    # Calibrate (optimize resources)
    ezkl.calibrate_settings(ONNX_PATH, SETTINGS_PATH, target="resources")
    print(f"[+] Calibrated settings at {SETTINGS_PATH}")


# ------------------------
# 3) Fetch SRS
# ------------------------
def fetch_srs():
    if os.path.exists(SRS_PATH):
        print(f"[i] Found existing {SRS_PATH}, skipping fetch.")
        return
    ezkl.get_srs(settings_path=SETTINGS_PATH, srs_path=SRS_PATH)
    print(f"[+] Downloaded SRS -> {SRS_PATH}")


# ------------------------
# 4) Compile circuit
# ------------------------
def compile_circuit():
    if os.path.exists(COMPILED_CIRCUIT):
        print(f"[i] Found existing {COMPILED_CIRCUIT}, skipping compile.")
        return
    ezkl.compile_circuit(
        model=ONNX_PATH,
        compiled_circuit=COMPILED_CIRCUIT,
        settings_path=SETTINGS_PATH,
    )
    print(f"[+] Compiled circuit -> {COMPILED_CIRCUIT}")


# ------------------------
# 5) Setup (generate proving / verification keys)
# ------------------------
def setup_keys():
    if os.path.exists(VK_PATH) and os.path.exists(PK_PATH):
        print(f"[i] Found existing keys, skipping setup.")
        return
    timed(
        "Setup",
        ezkl.setup,
        COMPILED_CIRCUIT,
        VK_PATH,
        PK_PATH,
        SRS_PATH,
    )
    print(f"[+] Setup complete: vk={VK_PATH}, pk={PK_PATH}")


# ------------------------
# 6) Generate witness
# ------------------------
def generate_witness():
    if os.path.exists(WITNESS_PATH):
        print(f"[i] Found existing {WITNESS_PATH}, skipping witness gen.")
        return
    ezkl.gen_witness(
        model=ONNX_PATH,
        compiled_circuit=COMPILED_CIRCUIT,
        input_data=INPUT_JSON,
        witness_path=WITNESS_PATH,
    )
    print(f"[+] Witness written -> {WITNESS_PATH}")


# ------------------------
# 7) Prove
# ------------------------
def generate_proof():
    if os.path.exists(PROOF_PATH):
        print(f"[i] Found existing {PROOF_PATH}, skipping proof gen.")
        return
    timed(
        "Prove",
        ezkl.prove,
        compiled_circuit=COMPILED_CIRCUIT,
        witness=WITNESS_PATH,
        pk_path=PK_PATH,
        proof_path=PROOF_PATH,
        srs_path=SRS_PATH,
    )
    print(f"[+] Proof generated -> {PROOF_PATH}")


# ------------------------
# 8) Verify
# ------------------------
def verify_proof():
    verified = ezkl.verify(proof_path=PROOF_PATH, vk_path=VK_PATH, srs_path=SRS_PATH)
    print(f"[+] Verified: {verified}")


# ------------------------
# Main
# ------------------------
def main():
    export_resnet18()
    generate_settings()
    fetch_srs()
    compile_circuit()
    setup_keys()
    generate_witness()
    generate_proof()
    verify_proof()


if __name__ == "__main__":
    main()
