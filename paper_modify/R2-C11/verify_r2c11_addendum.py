#!/usr/bin/env python3
"""Read-only verification of R2-C11 feature addendum against published Supplemental Data 1.

This command does not publish, relabel, retrain, overwrite either ZIP, or print ASINs.
Requires numpy and two locally available ZIP archives. Output is an aggregate-only JSON.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, re, sys, zipfile
from collections import Counter
from pathlib import Path
import numpy as np

ADDENDUM_FILES={
 '03_openclip512_features.npz', '04_interpretable36_features.csv',
 '05_development_folds.csv','README.md','PROVENANCE.json','SHA256SUMS.txt'
}
PU=('has_any_outer_imagery_observed', 'general_visual_appeal_observed_positive_core',
    'cute_friendly_observed_positive_core')
FOLDS={str(i):n for i,n in enumerate((837,828,828,828,822))}


def fail(msg): raise ValueError('R2-C11 cross-archive verification FAILED: '+msg)
def check(cond,msg):
    if not cond: fail(msg)
def digest(b): return hashlib.sha256(b).hexdigest().upper()
def file_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as src:
        for part in iter(lambda:src.read(1024*1024),b''): h.update(part)
    return h.hexdigest().upper()
def table(b):
    with io.StringIO(b.decode('utf-8-sig'),newline='') as f:
        reader=csv.DictReader(f)
        return (reader.fieldnames or []),list(reader)
def get_single(z,basename):
    names=[x for x in z.namelist() if not x.endswith('/') and x.split('/')[-1]==basename]
    check(len(names)==1,f'expected one {basename} in archive, found {len(names)}')
    return z.read(names[0])

def examine(addendum,supplemental,build_audit):
    report=json.loads(build_audit.read_text(encoding='utf-8-sig'))
    check(report.get('archive')==addendum.name,'candidate archive filename differs from original build audit')
    check(report.get('archive_bytes')==addendum.stat().st_size,'candidate archive byte size differs')
    check(file_sha(addendum)==report['archive_sha256'],'candidate ZIP byte SHA differs')
    check(report.get('development_products')==4143 and report.get('locked_test_feature_rows')==0,
          'unexpected original build populations')

    with zipfile.ZipFile(addendum) as az:
        check(az.testzip() is None,'candidate archive ZIP CRC failure')
        names=[s for s in az.namelist() if not s.endswith('/')]
        check(len(names)==6 and {n.split('/')[-1] for n in names}==ADDENDUM_FILES,
              'candidate ZIP member count/names unexpected')
        check(len({n.rsplit('/',1)[0] for n in names})==1,'candidate archive members span multiple directories')
        contents={p:get_single(az,p) for p in ADDENDUM_FILES}
        for name,raw in contents.items():
            check(digest(raw)==report['member_sha256'][name],f'candidate member SHA differs: {name}')
        with np.load(io.BytesIO(contents['03_openclip512_features.npz']),allow_pickle=False) as npz:
            check(set(npz.files)=={'parent_asin','openclip512'},'unexpected NPZ members')
            ids=npz['parent_asin'].tolist()
            X=npz['openclip512']
            check(X.shape==(4143,512) and X.dtype==np.float32 and np.isfinite(X).all(),
                  'OpenCLIP matrix shape/dtype/finiteness mismatch')
        check(len(ids)==4143 and len(set(ids))==4143 and all(isinstance(x,str) and x for x in ids),
              'NPZ product key count, uniqueness, or format mismatch')
        head36,raw36=table(contents['04_interpretable36_features.csv'])
        check(len(head36)==37 and head36[0]=='parent_asin' and len(set(head36))==37,
              'expected product ID + 36 unique feature columns')
        check(len(raw36)==4143 and [r['parent_asin'] for r in raw36]==ids,
              '36D features out of alignment with 512D IDs')
        for row in raw36:
            vals=[float(row[k]) for k in head36[1:]]
            check(len(vals)==36 and all(math.isfinite(v) for v in vals),'nonfinite 36D features')
            check(all(abs(v)<=1.00001 for v in vals[20:]),'OpenCLIP cosine outside [-1,1]')
        headfold,foldrows=table(contents['05_development_folds.csv'])
        check(headfold==['parent_asin','development_fold'] and len(foldrows)==4143,
              'fold file schema/row count mismatch')
        check([r['parent_asin'] for r in foldrows]==ids,'fold IDs do not align with 512D matrix')
        foldcount=Counter(r['development_fold'] for r in foldrows)
        check(dict(foldcount)==FOLDS,'five development fold sizes changed')
        fold_by_id={r['parent_asin']:r['development_fold'] for r in foldrows}
        p=json.loads(contents['PROVENANCE.json'])
        check(p['development_products']==4143 and p['heldout_products_in_addendum']==0,
              'ZIP provenance source population mismatch')
        checksum=contents['SHA256SUMS.txt'].decode('ascii')
        for name in ADDENDUM_FILES & set(report['member_sha256']) - {'SHA256SUMS.txt'}:
            if name in ('README.md','PROVENANCE.json'):continue
            check(f'{digest(contents[name])}  {name}' in checksum,'checksum file mismatch: '+name)

    with zipfile.ZipFile(supplemental) as sz:
        check(sz.testzip() is None,'published Supplemental Data 1 ZIP CRC failure')
        pub_head,pub_rows=table(get_single(sz,'01_modeling_ready_manifest.csv'))
    required={'parent_asin','split_partition','primary_response_sha256',*PU}
    check(required<=set(pub_head),'existing published manifest lacks necessary columns/PU outcomes')
    check(len(set(r['parent_asin'] for r in pub_rows))==len(pub_rows),'duplicate IDs in published manifest')
    pub_dev={r['parent_asin']:r for r in pub_rows if r['split_partition']=='development'}
    pub_heldout=sum(r['split_partition']=='locked_test' for r in pub_rows)
    if len(pub_dev)!=4143:
        return {'status':'BLOCKED_PUBLISHED_MANIFEST_LACKS_FULL_DEVELOPMENT_LABELS',
          'published_manifest_rows':len(pub_rows),'published_development_rows':len(pub_dev),
          'published_heldout_rows':pub_heldout,
          'action':'Full original development PU labels must be included in an approved supplemental release to reproduce P10 training; do not call R2-C11 complete.'}
    check(len(pub_rows)==5179 and pub_heldout==1036,'public population unexpected')
    check(set(pub_dev)==set(ids),'published development product IDs differ from feature addendum')
    positive={k:sum(int(pub_dev[asin][k]) for asin in ids) for k in PU}
    check([positive[k] for k in PU]==[186,120,68],'published development PU labels differ from frozen counts')
    img_folds={}
    for asin in ids:
        image=pub_dev[asin]['primary_response_sha256']
        check(bool(re.fullmatch('[0-9A-Fa-f]{64}',image)),'published image SHA format unexpected')
        fold=fold_by_id[asin]
        if image in img_folds:check(img_folds[image]==fold,'original image SHA group crosses development CV folds')
        img_folds[image]=fold
    return {'status':'PASS_FEATURE_ADDENDUM_LINKED_TO_PUBLISHED_DEVELOPMENT_LABELS',
        'development_rows':4143,'published_manifest_rows':len(pub_rows),
        'published_heldout_rows':pub_heldout,'published_development_rows':len(pub_dev),
        'five_original_grouped_fold_sizes':FOLDS,'development_observed_positives':positive,
        'feature_dimensions':{'OpenCLIP':512,'interpretable':36},
        'all_development_IDs_one_to_one_matched':True,
        'no_identical_primary_image_SHA_split_across_folds':True,
        'addendum_ZIP_SHA256':report['archive_sha256'],
        'published_Supplemental_Data_1_ZIP_SHA256':file_sha(supplemental),
        'source_ZIPs_not_modified':True,'public_deposit_not_established_by_this_check':True}

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--addendum',required=True,type=Path,help='Your new R2-C11 ZIP')
    cli.add_argument('--supplemental',required=True,type=Path,help='Existing PUBLIC Supplemental_Data_1_PeerJ.zip')
    cli.add_argument('--build-audit',required=True,type=Path,help='r2c11_addendum_build_audit.json')
    cli.add_argument('--report',required=True,type=Path,help='New JSON output path; never overwrites')
    args=cli.parse_args()
    check(not args.report.exists(),'refuse overwrite of prior report')
    result=examine(args.addendum,args.supplemental,args.build_audit)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(result['status'])
    if result['status'].startswith('BLOCKED'):sys.exit(2)
    print('Cross-ZIP integrity confirmed; no upload/publishing performed.')
if __name__=='__main__':main()
