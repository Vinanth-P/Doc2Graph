# V2 descriptive error audit — corrected

Source: `output/evaluation/handoff_learned_dev.json`
SHA-256: `a6e1246ea26d1b09048308ee9d0ebc6ab229dcbf6213d93c64ffe2bd9c569a36`
Documents: 500

Counts are computed from fact sets, not hard-coded. Samples are deterministic first examples within each stratum, not a random or human-adjudicated causal audit.

| Diagnostic | Count |
|---|---:|
| gold | 613 |
| predicted | 927 |
| tp | 47 |
| fp | 880 |
| fn | 566 |
| oracle_tp | 191 |
| oracle_fp | 587 |
| oracle_fn | 422 |
| fn_recovered_by_oracle | 148 |
| fn_also_missed_by_oracle | 418 |
| e2e_tp_missing_in_oracle | 4 |
| fp_both_unmatched | 255 |
| fp_aligned | 149 |
| fp_one_unmatched | 476 |
| fp_any_unmatched | 731 |

83.1% of scored false positives have at least one unmatched endpoint.

## Interpretation and corrections

- Unmatched means exact typed gold-cluster alignment failed or was ambiguous. Causes may include boundaries, types, spurious entities, missing annotations or incorrect resolution. It does not establish that the predicted relation is semantically correct.
- Oracle-recovered misses identify sensitivity to supplied mentions/clusters/contexts, not NER-only causation. End-to-end and oracle true-positive sets need not be nested; counts use intersections, not subtraction of totals.
- Misses remaining under oracle entities can involve candidate/schema coverage, context, thresholds, mapping, model errors or annotation issues. Their aggregate does not prove long-range reasoning is the cause.
- Candidate generation caps the DISTANCE feature at four. It does not exclude pairs farther than four sentences. The previous audit stated an incorrect cutoff.
- Pre-extraction merging exists in the learned/encoder branch. Its measured contribution requires traced merge examples or an ablation; code order alone does not prove how many facts were lost.
- Strictly unmatched predicted pairs were labeled zero by prior robust retraining. This is a real unknown-label hazard. Its causal contribution to recall collapse is not isolated without a controlled masked-label retraining comparison.
- A sigmoid/dot-product scorer is not inherently incapable of distinguishing negatives. The saved GLiREL probe contains bad predictions, but it does not show founder above 0.70 for every ordered pair. Calibration and integration must be tested independently.
- Claim-status defaults and source mapping require regression tests before building dataset preparation. Automatic taxonomy is not a substitute for manual adjudication.

## Samples requiring review

### fp_both_unmatched

- Document 3: Ramey Idriss; fact `('unmatched:69e61c21ce3a59ce58253a0b1e48742071c707e88a68fc2e61fc278151cc4330:2:140:157', 'OWNS', 'unmatched:69e61c21ce3a59ce58253a0b1e48742071c707e88a68fc2e61fc278151cc4330:2:0:6')`
  Evidence: Idriss was a musician in dance orchestras on radio and recordings and in films , and also wrote television scripts and special material for the Ritz Brothers , Eddie Cantor , Jimmy Durante and Marion Hutton .
  Causal adjudication: pending.
- Document 11: Mola di Bari; fact `('unmatched:dfb8a5a9b3c1153d926fb2a6831f83fcb7210610d2d3a7f2f0dc12e2d35a4e80:2:127:140', 'MEMBER_OF', 'unmatched:dfb8a5a9b3c1153d926fb2a6831f83fcb7210610d2d3a7f2f0dc12e2d35a4e80:2:105:119')`
  Evidence: Mola 's city center is its main piazza , Piazza XX Settembre near the port and it also boasts a church ( Chiesa Matrice , i.e. Mother Church ) dating back to the 13thcentury .
  Causal adjudication: pending.
- Document 12: Westmere, New York; fact `('unmatched:d1d4ee2935be80c0ee414025347d83d7d2f2733fb42460295478639b0158a38d:1:188:196', 'MEMBER_OF', 'unmatched:d1d4ee2935be80c0ee414025347d83d7d2f2733fb42460295478639b0158a38d:1:52:55')`
  Evidence: Since the 1970 Census a census - designated place ( CDP ) has been established with the name of Westmere for tabulating the population of what the census has defined as the boundaries for Westmere . Crossgates Mall , the Capital District 's largest shopping mall , is in Westmere 's northeastern corner .
  Causal adjudication: pending.
- Document 24: Pittsburgh Catholic; fact `('unmatched:d90495099dcfc5168826836d0461b2ec49b81dac1038ff13673b5658b03d6779:0:61:87', 'OWNS', 'unmatched:d90495099dcfc5168826836d0461b2ec49b81dac1038ff13673b5658b03d6779:10:0:23')`
  Evidence: The Pittsburgh Catholic is the weekly Catholic newspaper for the Roman Catholic Diocese of Pittsburgh , published for lay people and priests . As a result , the Pittsburgh Catholic is a conservative publication as well .
  Causal adjudication: pending.
- Document 24: Pittsburgh Catholic; fact `('unmatched:d90495099dcfc5168826836d0461b2ec49b81dac1038ff13673b5658b03d6779:10:0:23', 'SUBSIDIARY_OF', 'unmatched:d90495099dcfc5168826836d0461b2ec49b81dac1038ff13673b5658b03d6779:0:61:87')`
  Evidence: The Pittsburgh Catholic is the weekly Catholic newspaper for the Roman Catholic Diocese of Pittsburgh , published for lay people and priests . As a result , the Pittsburgh Catholic is a conservative publication as well .
  Causal adjudication: pending.

### fp_aligned

- Document 4: ELAM (Latin American School of Medicine) Cuba; fact `('4', 'SUBSIDIARY_OF', '1')`
  Evidence: Escuela Latinoamericana de Medicina ( ELAM ) , formerly Escuela Latinoamericana de Ciencias Médicas ( in Spanish ; in English : Latin American School of Medicine ( LASM ) , formerly Latin American School of Medical Sciences ) , is a major international medical school in Cuba and a prominent part of the Cuban healthcare system . Established in 1999 and operated by the Cuban government , ELAM has been described as possibly being the largest medical school in the world by enrollment with approximately 19,550 students from 110 countries reported as enrolled in 2013 .
  Causal adjudication: pending.
- Document 17: Carey Business School; fact `('0', 'OWNS', '7')`
  Evidence: The Carey Business School is accredited by the Association to Advance Collegiate Schools of Business ( AACSB ) . James Carey ( 1751 - 1834 ) , the namesake of the Carey Business School , is a relative to Johns Hopkins ( founder of Johns Hopkins University and Hospital ) , a co - founder of the Gilman School , and ancestor to several founding trustees of the university and hospital .
  Causal adjudication: pending.
- Document 17: Carey Business School; fact `('0', 'SUBSIDIARY_OF', '7')`
  Evidence: The Carey Business School is accredited by the Association to Advance Collegiate Schools of Business ( AACSB ) . James Carey ( 1751 - 1834 ) , the namesake of the Carey Business School , is a relative to Johns Hopkins ( founder of Johns Hopkins University and Hospital ) , a co - founder of the Gilman School , and ancestor to several founding trustees of the university and hospital .
  Causal adjudication: pending.
- Document 17: Carey Business School; fact `('7', 'OWNS', '0')`
  Evidence: The Carey Business School is accredited by the Association to Advance Collegiate Schools of Business ( AACSB ) . James Carey ( 1751 - 1834 ) , the namesake of the Carey Business School , is a relative to Johns Hopkins ( founder of Johns Hopkins University and Hospital ) , a co - founder of the Gilman School , and ancestor to several founding trustees of the university and hospital .
  Causal adjudication: pending.
- Document 17: Carey Business School; fact `('8', 'FOUNDED', '0')`
  Evidence: The Carey Business School is accredited by the Association to Advance Collegiate Schools of Business ( AACSB ) . James Carey ( 1751 - 1834 ) , the namesake of the Carey Business School , is a relative to Johns Hopkins ( founder of Johns Hopkins University and Hospital ) , a co - founder of the Gilman School , and ancestor to several founding trustees of the university and hospital .
  Causal adjudication: pending.

### fp_one_unmatched

- Document 6: Soccer Academy; fact `('1', 'OWNS', 'unmatched:08309b391cd65ba6efc2f7b6d70c9470d8e5d6e4fa00335299e292ac24897856:0:0:18')`
  Evidence: The Soccer Academy is a children ’s television show produced by Capistrano Films . The Soccer Academy " qualifies as children - friendly programming from an instructional and educational perspective , and has been issued the IE certificate by the United States for the shows educational value to children . The parent company , Capistrano Films , has offices located in both Los Angeles , California and Detroit , Michigan .
  Causal adjudication: pending.
- Document 6: Soccer Academy; fact `('1', 'OWNS', 'unmatched:08309b391cd65ba6efc2f7b6d70c9470d8e5d6e4fa00335299e292ac24897856:5:142:144')`
  Evidence: The Soccer Academy is a children ’s television show produced by Capistrano Films . The Soccer Academy " qualifies as children - friendly programming from an instructional and educational perspective , and has been issued the IE certificate by the United States for the shows educational value to children . The parent company , Capistrano Films , has offices located in both Los Angeles , California and Detroit , Michigan .
  Causal adjudication: pending.
- Document 6: Soccer Academy; fact `('unmatched:08309b391cd65ba6efc2f7b6d70c9470d8e5d6e4fa00335299e292ac24897856:0:0:18', 'OWNS', '1')`
  Evidence: The Soccer Academy is a children ’s television show produced by Capistrano Films . The Soccer Academy " qualifies as children - friendly programming from an instructional and educational perspective , and has been issued the IE certificate by the United States for the shows educational value to children . The parent company , Capistrano Films , has offices located in both Los Angeles , California and Detroit , Michigan .
  Causal adjudication: pending.
- Document 6: Soccer Academy; fact `('unmatched:08309b391cd65ba6efc2f7b6d70c9470d8e5d6e4fa00335299e292ac24897856:0:0:18', 'SUBSIDIARY_OF', '1')`
  Evidence: The Soccer Academy is a children ’s television show produced by Capistrano Films . The Soccer Academy " qualifies as children - friendly programming from an instructional and educational perspective , and has been issued the IE certificate by the United States for the shows educational value to children . The parent company , Capistrano Films , has offices located in both Los Angeles , California and Detroit , Michigan .
  Causal adjudication: pending.
- Document 7: ROKETSAN; fact `('0', 'OWNS', 'unmatched:5906a58605653369650f13639f9022d443054b48ea8c8de212834ad757fc5811:5:53:59')`
  Evidence: Roketsan is best known for its vast range of unguided rockets as well as laser and infrared guided missiles such as Cirit and UMTAS . The company also produces subsystems for Stinger and Rapier missiles and provides technology and engineering solutions for other integrated civilian and military platforms . Roketsan is the only Turkish company to have obtained CMMI/ DEV 3 ( Capability Maturity Model Integration – for Development ) approval for all its design and development processes .
  Causal adjudication: pending.
