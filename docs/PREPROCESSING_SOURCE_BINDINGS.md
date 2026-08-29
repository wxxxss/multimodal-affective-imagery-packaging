# Frozen preprocessing source bindings

This publication repository contains publication-safe copies of the upstream P1–P7 preprocessing, validation, and image-workflow source modules used for the manuscript. Their source identities are bound to the frozen internal research snapshot `1d72e3ba798f93d328c37bfe75b37032b9f21246`.

Except for the documented path-only sanitation of `scripts/metadata/screen_full_metadata_v2.py`, the public source modules below are byte-identical Git blobs to the frozen source snapshot. The sanitation removes a workstation-specific default path only; it does not alter scientific filtering logic, thresholds, ordering, or outputs.

Raw Amazon review text, raw third-party product images, image-source URLs, credentials, runtime logs, and internal administrative artifacts are not redistributed.

## Metadata eligibility screening

| Public path | Frozen historical path | Frozen Git blob SHA-1 |
| --- | --- | --- |
| `scripts/metadata/screen_full_metadata_v2.py` | `scripts/metadata/screen_full_metadata_v2.py` | `672495273e3b25ded76726d7472370b77fa6f44e` |
| `scripts/metadata/secondary_clean_candidates_v21.py` | `scripts/metadata/secondary_clean_candidates_v21.py` | `a120e06af0469dd19cb7515843be2a7f120d6389` |
| `scripts/metadata/secondary_clean_candidates_v22.py` | `scripts/metadata/secondary_clean_candidates_v22.py` | `f5da5eebd00505fb68f77143df153db5479ffd82` |

Public `screen_full_metadata_v2.py` has publication-safe path sanitation and therefore has public Git blob SHA-1 `fcb9ed289a6d1844d886396e85e19570bb88a26f`; the frozen historical blob above is retained as the scientific-source provenance binding.

## Review matching, text preprocessing, and label construction

| Public path | Frozen historical path | Frozen Git blob SHA-1 |
| --- | --- | --- |
| `scripts/reviews/match_reviews_to_valid_products.py` | `scripts/reviews/match_reviews_to_valid_products.py` | `5915c40d910994e06892978a72fc64188bdc1cd2` |
| `scripts/reviews/clean_and_extract_packaging_sentences.py` | `scripts/reviews/clean_and_extract_packaging_sentences.py` | `0c9dbfb3126118f44b50211201e04626758f2446` |
| `scripts/visual_packaging/strict_visual_packaging_classifier_v11.py` | `scripts/visual_packaging/strict_visual_packaging_classifier_v11.py` | `585db41e728dcc028944ffb0b30f4072512e5438` |
| `scripts/labels/build_affective_imagery_labels_v21.py` | `build_affective_imagery_labels_v21.py` | `1670e27305e26f1878c0f25dad0d0708e3e8ffcb` |

## Image acquisition, QA, and primary-image freeze

| Public path | Frozen historical path | Frozen Git blob SHA-1 |
| --- | --- | --- |
| `scripts/images/audit_p7_source_inventory.py` | `scripts/images/audit_p7_source_inventory.py` | `3e493824859aed84c5ee1f61d347ea88d87a2d5c` |
| `scripts/images/acquire_p7_primary_assets.py` | `scripts/images/acquire_p7_primary_assets.py` | `145a49ecc78700ff85942432b84b58c997951f5c` |
| `scripts/images/p7_c_primary_manifest_qa.py` | `scripts/images/p7_c_primary_manifest_qa.py` | `62c4bf41b84669150b1612da9a95f1b3708dcb95` |
| `scripts/images/p7_d_final_image_freeze.py` | `scripts/images/p7_d_final_image_freeze.py` | `7865b1eff54f4ac1b77dbda699b33c5266f1e6ed` |

The corresponding frozen public contracts are `config/image_assets/p7_source_contract.json`, `config/image_assets/p7_b_asset_contract.json`, `config/image_assets/p7_c_qa_contract.json`, `config/image_assets/p7_c_review_prompt.json`, and `config/image_assets/p7_d_final_freeze_contract.json`.

## Rule-guided validation/adjudication

| Public path | Frozen historical path | Frozen Git blob SHA-1 |
| --- | --- | --- |
| `scripts/validation/_prepare_affective_imagery_validation_v21_core.py` | `scripts/validation/_prepare_affective_imagery_validation_v21_core.py` | `eb07c3aa7dcf5b2fe842c6156cd4e5906503cc0f` |
| `scripts/validation/affective_imagery_action_error_mapping_v21.py` | `scripts/validation/affective_imagery_action_error_mapping_v21.py` | `9e2dcfef9db3c53c9be040a48d3aaf253c95c5bb` |
| `scripts/validation/affective_imagery_annotation_policy_v21.py` | `scripts/validation/affective_imagery_annotation_policy_v21.py` | `08365128aea46f9b6c2d7cdb356c2d34ffd0d70e` |
| `scripts/validation/affective_imagery_decision_sidecar_v21.py` | `scripts/validation/affective_imagery_decision_sidecar_v21.py` | `64287b666c3f6ea25503a6bf08c074232f600afa` |
| `scripts/validation/affective_imagery_final_adjudication_v21.py` | `scripts/validation/affective_imagery_final_adjudication_v21.py` | `2abcb544841108dce95ed81edd33501425e485bf` |
| `scripts/validation/affective_imagery_validation_manifest_v21.py` | `scripts/validation/affective_imagery_validation_manifest_v21.py` | `a10733fca7eec071d2e402ecbc216726aa8aeb65` |
| `scripts/validation/affective_imagery_validation_provenance_v21.py` | `scripts/validation/affective_imagery_validation_provenance_v21.py` | `4134a56fda6e9f06c40802d9a8e7e78a60376205` |
| `scripts/validation/normalize_affective_imagery_actions_v21.py` | `scripts/validation/normalize_affective_imagery_actions_v21.py` | `daf33b389f10aa75b6aa9707de6c1081f91e1934` |
| `scripts/validation/prepare_affective_imagery_validation_v21.py` | `scripts/validation/prepare_affective_imagery_validation_v21.py` | `3fdd739aa61f609470c3e94c7a38226432b928b7` |
| `scripts/validation/repair_affective_imagery_a2_rationales_v21.py` | `scripts/validation/repair_affective_imagery_a2_rationales_v21.py` | `70ad98efc5bf4ba2995f53ecc2455d9fd73ed973` |
| `scripts/validation/summarize_affective_imagery_validation_v21.py` | `scripts/validation/summarize_affective_imagery_validation_v21.py` | `79f96bbb6a363ec7f119182b6a97e52eca0c9e2d` |
| `scripts/validation/validate_affective_imagery_annotations_v21.py` | `scripts/validation/validate_affective_imagery_annotations_v21.py` | `813f39acf0f082bcaa1e0d15976dfd4ae50e50cd` |

The runtime action/error mapping contract is public at `config/affective_imagery/action_error_mapping_v21.json`, frozen Git blob SHA-1 `935880a607e9e4d93e9f77276c488cb4ede0f9e7`. There is no separate `annotation_policy_v21.json` in the frozen source snapshot; annotation-policy logic is implemented by `scripts/validation/affective_imagery_annotation_policy_v21.py`.

The model-assisted validation is reported as two independent rule-guided ChatGPT (GPT-5.5; OpenAI)-assisted review passes followed by model-assisted adjudication, not as independent human-gold annotation.

## Verification principle

For byte-identical public copies, the public Git blob SHA-1 should equal the frozen Git blob SHA-1 shown above. This is an integrity/provenance check, not a substitute for functional testing. The publication release also runs repository tests and CI to verify that required paths and documentation are present.
