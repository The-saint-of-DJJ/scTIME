# Zenodo Deposition Instructions

No Zenodo DOI has been minted for this revision. Do not cite or invent a DOI
before an author-controlled public deposit exists.

1. Verify the final code, license, documentation, figure/source-data manifest,
   and author-approved release contents.
2. Replace `REPLACE WITH VERIFIED DEPOSITOR NAME` in `zenodo.json` with the
   verified creator metadata. Add any additional verified creators there.
3. Create and push a versioned Git tag in the public scTIME repository.
4. Create the corresponding GitHub release and let the linked Zenodo account
   archive that release, or upload the release archive through Zenodo's web
   interface using `zenodo.json` as the metadata source.
5. Confirm the public landing page, version, license, creator list, and
   generated DOI before adding it to the manuscript's Code Availability text.
6. Preserve the release tag and DOI in the repository citation information.

The deposit should include code and redistributable derived/source tables only.
Do not upload external GEO, GDC, Xena, LinkedOmics, or CPTAC files unless their
licenses explicitly allow redistribution in that form.
