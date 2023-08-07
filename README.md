# MarkUs Exam Matcher

## Installation
It is recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html) to install this package, as it contains many pinned runtime dependencies.

### Production Installation (Recommended)
To install the production version of the package, run the following command in the console: 

```console
$ pip install git+https://github.com/MarkUsProject/markus-exam-matcher.git
```

### Branch Installation
To install the version specified by a particular branch (when permissions are granted), run the following command in the
console:
```console
$ pip install git+https://github.com/MarkUsProject/markus-exam-matcher.git@branch_name
```

### Commit Hash Installation
To install the version specified by a specific commit, the commit hash can be used by running the following command in
the console:
```console
$ pip install git+https://github.com/MarkUsProject/markus-exam-matcher.git@commit_hash
```
## Usage
This package works under the assumption that the characters to be detected are surrounded by boxes that are placed
side-by-side. An example image with this format is displayed below:

![Text](./examples/student_info_num.jpg?raw=true)

Below is a sample usage that generates a prediction from the file shown above, assuming it was titled `my_example.jpg`.
```console
$ python3 -m markus_exam_matcher ./my_example.jpg digit
0001250981
```
