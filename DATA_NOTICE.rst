Dataset Notice
==============

CaliBrain is distributed as software. The project license applies to the
source code in this repository and does not grant rights to third-party
datasets.

Some tutorials, workflows, or examples may be compatible with external
neuroimaging datasets, including CamCAN. Access to such datasets is governed
by the terms, conditions, and data-use agreements provided by the original
data owners.

In particular:

- users are responsible for obtaining any required permissions before
  downloading, accessing, processing, or sharing third-party data;
- users must comply with the applicable data-use agreement for any external
  dataset they choose to use with CaliBrain;
- CaliBrain does not redistribute third-party restricted datasets unless
  explicitly stated otherwise;
- references in the code or documentation to dataset names, file layouts, or
  processing scripts do not imply that redistribution rights are granted.

For CamCAN specifically, users should review the current terms and conditions
on the CamCAN data portal before any use:

https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/

If a workflow requires CamCAN or another restricted dataset, the recommended
pattern is that users obtain the data themselves under the relevant agreement
and then run the CaliBrain scripts locally on their own authorized copy.

Derived CamCAN example data
---------------------------

The CaliBrain-data repository provides the forward solutions of four CamCAN participants (``CC120166``, ``CC120264``, ``CC120309``, ``CC120313``) as derived example data (``fetch_example_data("camcan")``).
They contain the source-space geometry and the sensor gain matrix computed for these
participants, and no images and no participant information beyond the pseudonymous
participant IDs. They remain subject to the CamCAN terms: if you use them, acknowledge
CamCAN and cite the CamCAN publications as described on the data portal. They do not
replace requesting access to the CamCAN data if you need the original recordings.
