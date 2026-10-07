import sys, os
import numpy as np
import pyarrow.parquet as pq

k, K = int(sys.argv[1]), int(sys.argv[2])
src = '/9950backfile/zhangem/data/dolma3_multiscale/dolma3_mix_sample_42B.parquet'
outdir = '/9950backfile/zhangem/data/dolma3_1025_42B_packed'
os.makedirs(outdir, exist_ok=True)
dst = os.path.join(outdir, f'{k:03d}_00000.parquet.data')
if os.path.exists(dst) and os.path.getsize(dst) > 0:
    print(f'[shard {k}] exists, skip', flush=True); sys.exit(0)
tmp = dst + '.partial'
pf = pq.ParquetFile(src)
n_rg = pf.metadata.num_row_groups
per = (n_rg + K - 1) // K
rgs = list(range(k * per, min((k + 1) * per, n_rg)))
EOS = np.array([100257], dtype='<u4')
docs = tokens = 0
buf = []
with open(tmp, 'wb') as out:
    for batch in pf.iter_batches(batch_size=512, columns=['input_ids'], row_groups=rgs):
        for ids in batch.column(0):
            a = np.asarray(ids, dtype='<u4')
            buf.append(a); buf.append(EOS)
            docs += 1; tokens += a.shape[0] + 1
            if tokens % 100_000_000 < 600:
                pass
        if len(buf) > 400_000:
            np.concatenate(buf).tofile(out); buf = []
    if buf:
        np.concatenate(buf).tofile(out)
os.replace(tmp, dst)
print(f'[shard {k}] DONE docs={docs:,} tokens={tokens:,} file={dst}', flush=True)
