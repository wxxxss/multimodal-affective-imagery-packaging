# Frozen preprocessing source bindings

This publication repository is organized around the manuscript-facing machine-learning workflow. The raw-source preprocessing implementation remains bound to the frozen internal research snapshot `1d72e3ba798f93d328c37bfe75b37032b9f21246` and is documented here for provenance.

The historical scripts were written for the author's local research workspace and some contain workstation-specific default paths that are overridden through command-line arguments. They are not required to recompute the publication-facing held-out metrics from PeerJ Supplemental Data 1 and are therefore not copied wholesale into this clean release.

## Metadata eligibility screening

| Historical path | Frozen Git blob SHA-1 |
| --- | --- |
| `scripts/metadata/screen_full_metadata_v2.py` | `672495273e3b25ded76726d7472370b77fa6f44e` |
| `scripts/metadata/secondary_clean_candidates_v21.py` | `a120e06af0469dd19cb7515843be2a7f120d6389` |
| `scripts/metadata/secondary_clean_candidates_v22.py` | `f5da5eebd00505fb68f77143df153db5479ffd82` |

## Review matching and text preprocessing

| Historical path | Frozen Git blob SHA-1 |
| --- | --- |
| `scripts/reviews/match_reviews_to_valid_products.py` | `5915c40d910994e06892978a72fc64188bdc1cd2` |
| `scripts/reviews/clean_and_extract_packaging_sentences.py` | `0c9dbfb3126118f44b50211201e04626758f2446` |
| `scripts/visual_packaging/strict_visual_packaging_classifier_v11.py` | `585db41e728dcc028944ffb0b30f4072512e5438` |
| `build_affective_imagery_labels_v21.py` | `1670e27305e26f1878c0f25dad0d0708e3e8ffcb` |

## Image acquisition, QA, and primary-image freeze

| Historical path | Frozen Git blob SHA-1 |
| --- | --- |
| `scripts/images/audit_p7_source_inventory.py` | `3e493824859aed84c5ee1f61d347ea88d87a2d5c` |
| `scripts/images/acquire_p7_primary_assets.py` | `145a49ecc78700ff85942432b84b58c997951f5c` |
| `scripts/images/p7_c_primary_manifest_qa.py` | `62c4bf41b84669150b1612da9a95f1b3708dcb95` |
| `scripts/images/p7_d_final_image_freeze.py` | `7865b1eff54f4ac1b77dbda699b33c5266f1e6ed` |

Historical image-contract blobs: `978e2e17de14eb0b4668d46281bd3c969108c22e`, `351676d73960fa90b0b6febd5d7189a5be699874`, `1085853f641939edffb5d990dc326d5cbb781e74`, `5ac6a63cdd9ebd4675de8ed6bf28112a9c03f560`, and `8767a9752a9e8ef672e097ca077d287ddde883cc`.

## Rule-guided validation/adjudication

The publication release contains the validation protocol and annotation schema under `docs/validation/`. The frozen implementation was composed of the following tracked modules:

| Historical module | Frozen Git blob SHA-1 |
| --- | --- |
| `_prepare_affective_imagery_validation_v21_core.py` | `eb07c3aa7dcf5b2fe842c6156cd4e5906503cc0f` |
| `affective_imagery_action_error_mapping_v21.py` | `9e2dcfef9db3c53c9be040a48d3aaf253c95c5bb` |
| `affective_imagery_annotation_policy_v21.py` | `08365128aea46f9b6c2d7cdb356c2d34ffd0d70e` |
| `affective_imagery_decision_sidecar_v21.py` | `64287b666c3f6ea25503a6bf08c074232f600afa` |
| `affective_imagery_final_adjudication_v21.py` | `2abcb544841108dce95ed81edd33501425e485bf` |
| `affective_imagery_validation_manifest_v21.py` | `a10733fca7eec071d2e402ecbc216726aa8aeb65` |
| `affective_imagery_validation_provenance_v21.py` | `4134a56fda6e9f06c40802d9a8e7e78a60376205` |
| `normalize_affective_imagery_actions_v21.py` | `daf33b389f10aa75b6aa9707de6c1081f91e1934` |
| `prepare_affective_imagery_validation_v21.py` | `3fdd739aa61f609470c3e94c7a38226432b928b7` |
| `repair_affective_imagery_a2_rationales_v21.py` | `70ad98efc5bf4ba2995f53ecc2455d9fd73ed973` |
| `summarize_affective_imagery_validation_v21.py` | `79f96bbb6a363ec7f119182b6a97e52eca0c9e2d` |
| `validate_affective_imagery_annotations_v21.py` | `813f39acf0f082bcaa1e0d15976dfd4ae50e50cd` |

The model-assisted validation is reported as two independent rule-guided ChatGPT (GPT-5.5; OpenAI)-assisted review passes followed by model-assisted adjudication, not as independent human-gold annotation.

## Publication boundary

These bindings provide an auditable connection to the frozen source snapshot without modifying the historical research code or exposing unnecessary workstation-specific implementation details. Scientific values reported in the manuscript are reproduced from the frozen downstream artifacts and the publication-facing correction overlay described in `docs/AUROC_AUDIT.md`.
