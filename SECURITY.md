# Security and deployment scope

## Reporting a vulnerability

Do not post credentials, private dataset locations, or exploitable host details in a public issue.
Contact the repository owner through the email listed in `CITATION.cff` with a minimal reproducer and
the affected commit. Ordinary correctness bugs and reproducibility failures belong in GitHub Issues.

## Scope

This is an academic simulation artifact. It is not safety-certified for physical robots, industrial
automation, or unattended actuation. The LIBERO action bridge, camera mapping, normalization, and
control mode are part of the evaluated policy contract; changing any of them invalidates the reported
behavior until a new causal probe and registered evaluation are completed.

Never store GitHub tokens, Hugging Face tokens, private model weights, or raw user data in this
repository. The release gate checks evidence integrity but is not a substitute for deployment risk
assessment.
