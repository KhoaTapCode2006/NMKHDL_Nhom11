# NMKHDL_Nhom11

## 5. Recommended processing workflow

A clean and maintainable pipeline is as follows:

1. Load the assigned arXiv IDs.
2. Resolve each ID with the arXiv API to obtain the latest version and metadata.
3. Determine the number of paper versions and download each versioned source archive.
4. Filter the archive to keep only `.tex` and `.bib` files.
5. Save the files in a versioned folder structure with metadata JSON output.
6. Query Semantic Scholar once per base paper to retrieve the references.
7. Store reference results in JSON or structured database tables.
8. Validate the output directories and check for missing or empty source archives.

This workflow is robust, easy to debug, and aligned with the course requirement of building a crawler that respects protocol rules while producing usable research data.


## 6. Comparison of the two crawler strategies

The project uses two complementary approaches: the arXiv crawler and the OAI-based crawler. They overlap in purpose but differ in how they access and harvest data.

| Aspect | arXiv crawler | OAI-PMH crawler |
| --- | --- | --- |
| Primary use | Targeted paper discovery and versioned source download | Metadata harvesting and broader repository-style access |
| Data source | `arxiv.py` + `export.arxiv.org` | `oaipmh.arxiv.org/oai` |
| Query style | Direct ID or keyword lookup | Record harvesting with `ListRecords` and `GetRecord` |
| Version handling | Explicitly downloads each version (`v1`, `v2`, `v3`) | Can harvest metadata at the record level and track revisions |
| Source data | Downloads `.tar.gz` paper source archives | Can also be combined with source download logic |
| Best use case | Known list of paper IDs | Large or date-based metadata collection |
| Rate limits | API-based pacing and retries | OAI request throttling and pagination handling |
| Complexity | Lower for targeted paper sets | Higher for scalable metadata harvesting |
| Output | Paper metadata, version folders, references | Structured metadata records, identifiers, harvesting state |

This table is a practical sample comparison. In practice, both crawler types can be used together to achieve precise control over a target set and broader repository visibility.
