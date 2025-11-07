# Transaction CSV Schema (synthetic/anonymized)

transaction_id,branch,client_id,client_name,transaction_date,amount,currency,narrative,label
TXN0001,WestBranch,CL001,John Doe,2024-10-12,4800000,KES,"Transferred KES 4,800,000 to foreign account in Dubai; no updated KYC documents",Non-Compliant

## Notes
- `client_name` may be synthetic for demos; remove if using strict anonymization.
- Labels for training: Compliant, KYC Violation, STR Missing, High-Risk, Other.
