# Datasets

The datasets are from Pison and can be downloaded from https://drive.google.com/drive/folders/1KQ1DjvIWpHikOg1JgmjlSWM3aAlvq-h7?usp=sharing.

They should be pre-processed as such:

```shell
dos2unix *.json
gawk -i inplace 'NF' *.json
```

This will make sure that all JSON lines are separated by exactly 1 LF, rather than multiple CRLF.

An additional file should also be created to make the index of the file fall within the signed 32-bit integer
index size limit. This is used for the sequential benchmark, which has not been implemented for larger files.

```shell
head -n -30000 twitter_small_records.json > twitter_small_records_smaller.json
```
