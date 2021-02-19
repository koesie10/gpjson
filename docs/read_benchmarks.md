# Java
Reading 2.7GB file with 1MB buffer: 700-800 MB/s
Reading 2.7GB file with 1GB buffer: 700-800 MB/s

Reading 2.7GB file with 1MB buffer (direct pointer, managed memory): 1.9 - 2.4 GB/s
Reading 2.7GB file with 1MB buffer (direct pointer, unmanaged memory): 4.6 - 5.0 GB/s

# Raw
```shell
time dd if=arxiv-metadata-oai-snapshot.json of=/dev/null bs=8k
```

4k = 3.0 GB/s
8k: 4.7 GB/s
1024k: 7.7 GB/s
