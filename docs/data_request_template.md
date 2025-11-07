# Data Request Template (for anonymized internal samples)

**Project:** NLP-Based Early Warning System for Regulatory Compliance (Kenyan Banking)  
**Requester:** Melanie Mweru Wachira  
**Purpose:** Academic prototype to detect potential non-compliance in AML/KYC narratives.

## Requested Fields (sample/anonymized)
- transaction_id (randomized)
- branch (generalized or code)
- client_type (e.g., retail/SME)
- transaction_date (yyyy-mm-dd, +/- jitter)
- amount (bucket or perturbed)
- currency (KES)
- narrative (text, redact PII)
- label (if available): Compliant / KYC Violation / STR Missing / High-Risk / Other

## Safeguards
- No direct identifiers (names, IDs, phone, account numbers).
- Replace with synthetic or hashed tokens.
- Storage: encrypted at rest; access limited to research account.
- Retention: delete within 12 months or upon project completion.

## Legal/Ethics
- Comply with Kenya Data Protection Act (2019).
- Use only for research; no re-distribution.
