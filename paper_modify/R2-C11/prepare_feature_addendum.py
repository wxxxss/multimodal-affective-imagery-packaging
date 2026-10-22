#!/usr/bin/env python3
"""R2-C11: build a reviewer-facing DEVELOPMENT feature addendum, locally only.

DO NOT upload the zip without separately reviewing its contents and applicable
third-party/source constraints. Copies no raw images or reviews, does not touch
P8-P12 sources, and does not train or evaluate any models.

Required: P8 frozen manifest; P9 4994x512 matrix, 36-feature CSV, product map,
P10 frozen contract, P9 descriptor schema. The resulting feature rows are keyed
by the exact parent_asin already disclosed by original Supplemental Data 1.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

RELEASE_NAME = "R2-C11_development_features_addendum_v1"
ORIGINAL_PUBLISHED_MANIFEST_SHA = "5810EA486657D06C59DB63F374B57543773658B250DD1A08B2B3CF6B3E97BCAE"
P8_REL="data/processed/modeling_readiness_p8_5180/p8_b_modeling_ready_split/01_modeling_ready_manifest.csv"
P9_BASE="data/processed/modeling_features_p9_5180/p9_visual_feature_freeze"
P9_REL={
    "embeddings":P9_BASE+"/03_openclip_image_embeddings.npy",
    "features":P9_BASE+"/04_interpretable_image_features.csv",
    "product_map":P9_BASE+"/05_product_feature_map.csv",
    "inventory":P9_BASE+"/02_unique_image_inventory.csv",
}
C_REL="config/modeling/p10_development_model_contract.json"
SCHEMA_REL="config/modeling/p9_interpretable_feature_schema.json"
OUTPUTS=("03_openclip512_features.npz","04_interpretable36_features.csv",
         "05_development_folds.csv")


def check(value, message):
    if not value:
        raise ValueError("R2-C11 feature addendum: "+message)


def read_json(path):
    with path.open("r",encoding="utf-8-sig") as f:
        return json.load(f)


def rows_csv(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        r=csv.DictReader(f)
        result=list(r)
        return tuple(r.fieldnames or ()),result


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest().upper()


def sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):
            h.update(b)
    return h.hexdigest().upper()


def bytes_csv(fields, rows):
    obj=io.StringIO(newline="")
    w=csv.DictWriter(obj,fieldnames=fields,extrasaction="raise",lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return obj.getvalue().encode("utf-8")


def input_paths(root):
    d={"p8":root/P8_REL,"contract":root/C_REL,"schema":root/SCHEMA_REL}
    d.update({name:root/path for name,path in P9_REL.items()})
    return d


def preflight(paths,out):
    missing=[str(v) for v in paths.values() if not v.is_file()]
    check(not missing,"missing frozen inputs: "+"; ".join(missing))
    check(not out.exists(),"will not overwrite existing staging directory: "+str(out))


def verify_existing_public_manifest(path,dev_ids):
    """Optional match to already published Supplemental Data 1, not its scores."""
    if path is None:
        return {"checked":False,"explanation":"Optional original Supplemental Data 1 manifest not supplied"}
    check(path.is_file(),"existing Supplemental Data 1 manifest path not found")
    check(sha_file(path)==ORIGINAL_PUBLISHED_MANIFEST_SHA,
          "published Supplemental Data 1 source SHA does not match expected frozen export")
    header,rr=rows_csv(path)
    check(len(rr)==5179 and "parent_asin" in header and "split_partition" in header,
          "published manifest has unexpected schema/count")
    pub_dev={r["parent_asin"] for r in rr if r["split_partition"]=="development"}
    check(len(pub_dev)==4143 and pub_dev==set(dev_ids),
          "published Supplemental Data 1 has different development ASINs")
    return {"checked":True,"sha256":sha_file(path),"development_product_set_match":True}


def prepare(paths,out,public_manifest=None):
    preflight(paths,out)
    contract=read_json(paths["contract"])
    schema=read_json(paths["schema"])
    check(contract.get("contract_version")=="p10_v1.0","original model contract version changed")
    check(contract["development_population"]["development_rows"]==4143,
          "original development population changed")
    check(contract["development_population"]["locked_test_rows"]==1036,
          "original held-out population changed")
    check(contract["development_population"]["group_key"]=="primary_response_sha256",
          "original CV image grouping changed")
    need={"p8":contract["upstream"]["p8_b_formal_sha256"]["01_modeling_ready_manifest.csv"]}
    for name,p in P9_REL.items():
        need[name]=contract["upstream"]["p9_formal_sha256"][p.rsplit("/",1)[-1]]
    for key,want in need.items():
        check(sha_file(paths[key])==want,"historical SHA mismatch: "+key)
    classical=[x["name"] for x in schema["classical_features"]]
    semantic=[x["name"] for x in schema["semantic_features"]]
    desc=classical+semantic
    check(len(classical)==20 and len(semantic)==16 and len(set(desc))==36,
          "feature schema is not 20 + 16 unique columns")

    _,p8=rows_csv(paths["p8"])
    _,products=rows_csv(paths["product_map"])
    head,features=rows_csv(paths["features"])
    _,inventory=rows_csv(paths["inventory"])
    check(head==tuple(["feature_row_index","image_sha256",*desc]),
          "frozen P9 36D CSV header/order changed")
    check(len(products)==5180 and len(features)==len(inventory)==4994,
          "P9 source row count changed")
    by_asin={r["parent_asin"]:r for r in products}
    check(len(by_asin)==5180,"duplicate ASIN in P9 product map")
    primary_map=[]
    locked=0
    for r in p8:
        if r["main_analysis_included"]!="true":
            continue
        part=r["split_partition"]
        if part=="locked_test":
            locked+=1
            continue  # do not access test labels or features
        check(part=="development","unexpected partition in P8")
        p=by_asin.get(r["parent_asin"])
        check(p is not None and p["primary_feature_available"]=="1",
              "missing frozen primary image mapping")
        check(p["primary_response_sha256"]==r["primary_response_sha256"],
              "P8/P9 primary image SHA mismatch")
        idx=int(p["primary_feature_row_index"])
        check(0<=idx<4994,"P9 image index out of range")
        check(inventory[idx]["image_sha256"]==r["primary_response_sha256"] and
              features[idx]["image_sha256"]==r["primary_response_sha256"] and
              int(inventory[idx]["feature_row_index"])==idx and
              int(features[idx]["feature_row_index"])==idx,
              "P9 image / feature row misalignment")
        fold=int(r["development_fold"])
        check(fold in range(5),"P8 development fold outside 0..4")
        primary_map.append((int(r["input_order"]),r["parent_asin"],idx,fold))
    primary_map.sort()
    check(len(primary_map)==4143 and locked==1036,"P8 population changed")
    ids=[r[1] for r in primary_map]
    check(len(set(ids))==4143,"duplicate development ASIN")
    check(Counter(r[3] for r in primary_map)==Counter({0:837,1:828,2:828,3:828,4:822}),
          "original five development folds changed")

    public_audit=verify_existing_public_manifest(public_manifest,ids)
    matrix=np.load(paths["embeddings"],allow_pickle=False)
    check(matrix.shape==(4994,512) and matrix.dtype==np.dtype("float32"),
          "frozen P9 numpy embedding shape/dtype changed")
    check(np.isfinite(matrix).all(),"original 512D embeddings contain nonfinite values")
    idxs=np.asarray([r[2] for r in primary_map],dtype=np.int64)
    selected=np.ascontiguousarray(matrix[idxs])
    selected_id=np.asarray(ids,dtype="<U"+str(max(len(x) for x in ids)))

    # Stage in memory: output archive uses identical original source precision.
    out_npz=io.BytesIO()
    np.savez_compressed(out_npz,parent_asin=selected_id,openclip512=selected)
    npz_data=out_npz.getvalue()
    feature_csv=bytes_csv(["parent_asin",*desc],(
        {"parent_asin":asin,**{name:features[index][name] for name in desc}}
        for _,asin,index,_ in primary_map))
    fold_csv=bytes_csv(["parent_asin","development_fold"],(
        {"parent_asin":asin,"development_fold":fold}
        for _,asin,_,fold in primary_map))

    # Reconstruct the exact released matrices *before writing a ZIP*.
    with np.load(io.BytesIO(npz_data),allow_pickle=False) as ar:
        check(set(ar.files)=={"parent_asin","openclip512"},"NPZ members mismatch")
        check(np.array_equal(ar["openclip512"],selected) and
              np.array_equal(ar["parent_asin"],selected_id),"NPZ roundtrip mismatch")
    reread=list(csv.DictReader(io.StringIO(feature_csv.decode("utf8"))))
    check(len(reread)==4143 and all(reread[i]["parent_asin"]==ids[i]
          for i in range(4143)),"36D CSV row identity mismatch")
    for i,r in enumerate(reread):
        orig=features[primary_map[i][2]]
        for col in desc:
            check(math.isfinite(float(r[col])) and r[col]==orig[col],
                  "36D original numeric source mismatch: "+col)
    foldrows=list(csv.DictReader(io.StringIO(fold_csv.decode("utf8"))))
    check(len(foldrows)==4143 and all(
          foldrows[i]["parent_asin"]==ids[i] and
          int(foldrows[i]["development_fold"])==primary_map[i][3]
          for i in range(4143)),"original development fold binding changed")

    data=dict(zip(OUTPUTS,(npz_data,feature_csv,fold_csv)))
    hashes={k:sha_bytes(v) for k,v in data.items()}
    provenance={
      "study":"PeerJ cs-147682","artifact":"R2-C11 development-only feature addendum",
      "status":"LOCAL_STAGING_ONLY_AUTHOR_RELEASE_DECISION_PENDING",
      "development_products":4143,"heldout_products_in_addendum":0,
      "original_modeling_products":5179,"original_frozen_P9_unique_images":4994,
      "features":{"OpenCLIP":512,"interpretable":36,"classical":20,"semantic_OpenCLIP_cosine":16},
      "original_development_fold_counts":{"0":837,"1":828,"2":828,"3":828,"4":822},
      "row_key":"parent_asin (matches originally published Supplemental Data 1)",
      "public_manifest_comparison":public_audit,
      "source_SHA256":{k:sha_file(p) for k,p in paths.items()},
      "added_file_SHA256":hashes,
      "raw_images_or_review_text_included":False,
      "observed_positive_labels_duplicated":False,
      "original_P8_P12_files_modified":False,
      "new_training_or_locked_test_evaluation":False,
      "rights_or_public_release_authorized_by_technical_audit":False,
    }
    readme=f"""# R2-C11 — Original feature matrices for development-stage reproduction

Manuscript cs-147682; researcher-generated representations derived from Amazon Reviews'23 product-image sources. Version: local release candidate, not yet deposited.

This addendum works together with the **already published** Supplemental Data 1 / 01_modeling_ready_manifest.csv (5,179 products, parent_asin, frozen train/test partition and three PU observed-positive labels). Existing frozen held-out scores remain in Supplemental Data 1 / 02_locked_test_predictions.csv. No labels or held-out scores are duplicated in this addendum.

Contents:
- 03_openclip512_features.npz: a NumPy NPZ archive containing `parent_asin` (4,143 Unicode product identifiers) and `openclip512` (4,143 × 512 float32 vectors). Load via `np.load(path, allow_pickle=False)`.
- 04_interpretable36_features.csv: 4,143 rows keyed by `parent_asin`, followed by 36 original P9 columns in the frozen original order: 20 classical raster statistics and 16 OpenCLIP-based cosine similarities.
- 05_development_folds.csv: 4,143 product IDs with `development_fold` 0–4. This is required to reconstruct the original fixed five-fold hyperparameter search because the prior public Supplemental Data 1 omitted that fold field.
- PROVENANCE.json: source and release SHA-256 values, data counts, reproducibility boundaries.
- SHA256SUMS.txt: integrity values for the above three numerical data files.

To replay original development training, join the two feature tables and 05_development_folds.csv against published 01_modeling_ready_manifest.csv by `parent_asin` (one-to-one). Use only rows with `split_partition == development`. The original P10 training code performs within-training-fold StandardScaler and an unweighted LogisticRegression with lbfgs, max_iter=5000, tol=1e-8, random_state=20260818, and originally selected C from {{0.01,0.1,1,10,100}} on five original grouped folds. 1 means observed positive; 0 is PU-unlabeled, not a confirmed negative.

**Limit:** This addendum provides the original development feature matrices and exact CV assignments. It does NOT contain 1,036 original locked-test feature rows; re-running raw test prediction generation from features therefore still requires those held-out features. The previously released frozen held-out scores remain sufficient to recompute original published held-out metrics. It also does NOT contain original image files, consumer reviews, product image URLs, or claim individual image-redistribution rights. Publication should follow the author's actual source-terms and content review.

Generated with paper_modify/R2-C11/prepare_feature_addendum.py; original P8–P12 files were read only.
""".encode("utf-8")
    p_bytes=json.dumps(provenance,ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8")+b"\n"
    checksum_text="".join(f"{hashes[name]}  {name}\n" for name in OUTPUTS).encode("ascii")
    entries={**data,"README.md":readme,"PROVENANCE.json":p_bytes,"SHA256SUMS.txt":checksum_text}

    # Final archive is a new isolated directory; no source modification/overwrite.
    out.mkdir(parents=True,exist_ok=False)
    archive=out/(RELEASE_NAME+".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,content in entries.items():z.writestr(RELEASE_NAME+"/"+name,content)
    with zipfile.ZipFile(archive) as z:
        check(z.testzip() is None,"output ZIP failed CRC inspection")
        for name,expected_data in entries.items():
            check(z.read(RELEASE_NAME+"/"+name)==expected_data,
                  "ZIP archive member differs from staged source "+name)
    audit={"status":"LOCAL_RELEASE_ADDENDUM_BUILT_NOT_UPLOADED",
           "archive":archive.name,"archive_bytes":archive.stat().st_size,
           "archive_sha256":sha_file(archive),
           "development_products":4143,"locked_test_feature_rows":0,
           "member_sha256":{name:sha_bytes(payload) for name,payload in entries.items()},
           "published_Supplemental_Data_1_manifest_verified":public_audit["checked"],
           "rights_approval_and_public_deposit":False,
           "original_frozen_source_files_modified":False,
           "release_notes":"Review contents and source terms before uploading; do not claim previously released."}
    (out/"r2c11_addendum_build_audit.json").write_text(
        json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ADDENDUM BUILT AND VERIFIED (LOCAL ONLY):",archive)
    print("DEVELOPMENT: 4143; locked-test features in archive: 0")
    print("ZIP SHA256:",audit["archive_sha256"])
    print("Review publication scope before posting on Zenodo or any other website.")
    print("To independently check package build, upload ONLY r2c11_addendum_build_audit.json here.")


def self_test():
    check(sum((837,828,828,828,822))==4143,"original fold size invariant")
    check(len(set(OUTPUTS))==3,"added data files")
    test={"parent_asin":np.asarray(["TEST-ASIN"],dtype="<U9"),
          "openclip512":np.ones((1,512),dtype=np.float32)}
    bio=io.BytesIO()
    np.savez_compressed(bio,**test)
    with np.load(io.BytesIO(bio.getvalue()),allow_pickle=False) as ar:
        check(np.array_equal(ar["openclip512"],test["openclip512"]),"NPZ float32 roundtrip")
        check(ar["parent_asin"].tolist()==["TEST-ASIN"],"NPZ identifier roundtrip")
    m=bytes_csv(["parent_asin","development_fold"],[{"parent_asin":"TEST-ASIN","development_fold":0}])
    check(len(list(csv.DictReader(io.StringIO(m.decode("utf8")))))==1,"CSV roundtrip")
    print("SELF-TEST PASS: frozen folds, NPZ roundtrip and CSV integrity")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    for option in ("self-test","preflight","build"):
        group.add_argument("--"+option,action="store_true")
    ap.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument("--output-dir",type=Path)
    ap.add_argument("--published-manifest",type=Path,help="Optional extracted 01_modeling_ready_manifest.csv from existing published Supplemental Data 1")
    args=ap.parse_args()
    if args.self_test:
        return self_test()
    root=args.repo_root.resolve()
    out=args.output_dir.resolve() if args.output_dir else root/"paper_modify/R2-C11/local_feature_addendum"
    paths=input_paths(root)
    preflight(paths,out)
    if args.preflight:
        print("PREFLIGHT PASS: original read-only inputs present; output does not exist.")
        if args.published_manifest:
            print("Published manifest was supplied; its SHA/product IDs will be verified during --build.")
        else:
            print("Optional published Supplemental Data 1 manifest not supplied. IDs still use original frozen P8 ASIN.")
        return
    prepare(paths,out,args.published_manifest)

if __name__=="__main__":main()
