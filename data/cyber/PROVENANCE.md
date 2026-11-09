# Data provenance: NSL-KDD

## Source

NSL-KDD (Tavallaee, Bagheri, Lu and Ghorbani, 2009), the cleaned version of
the KDD Cup 1999 network intrusion data released by the Canadian Institute
for Cybersecurity, University of New Brunswick. Each record is one network
connection described by 41 features and labelled as normal traffic or as
one of many attack types.

The files `KDDTrain+.txt.gz` and `KDDTest+.txt.gz` are the unmodified
`KDDTrain+.txt` and `KDDTest+.txt`, gzip-compressed. They were obtained from
the public GitHub mirror <https://github.com/defcom17/NSL_KDD> via
`git clone`, because the original UNB download page is not reachable from
the environment this project was built in. This is stated here rather than
describing the files as downloaded from UNB directly.

## How the project uses it

* `KDDTrain+` is split into a training part (AI and analyst cue model) and an
  optimisation part (protocol design). `KDDTest+` is the held-out
  evaluation set. It contains attack types that never appear in training,
  which is why it also serves as a test under distribution shift.
* Attacks make up roughly half of NSL-KDD, which is unrealistic for alert
  triage. Attacks are subsampled with a fixed seed to 20% of each split
  (see `hai/domains/cyber.py`).
* The AI sees the basic connection features and traffic statistics. The
  13 "content" features, which in the original KDD design are derived from
  inspecting the session payload on the host, are withheld from the AI and
  form the basis of the simulated analyst's judgment.

## Known limitations of the dataset

NSL-KDD is old (derived from 1998 DARPA traffic) and its traffic is
synthetic. It is widely used as a benchmark, not as a picture of modern
networks. It is used here as a controlled testbed for interaction protocols,
not to claim anything about intrusion detection performance in practice.

## Citation

M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani. A Detailed Analysis of the
KDD CUP 99 Data Set. IEEE Symposium on Computational Intelligence for
Security and Defense Applications (CISDA), 2009.
