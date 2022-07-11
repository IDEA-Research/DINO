import os
import numpy as np
import base64
import cv2
from tqdm import tqdm
import yaml


def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.frombuffer(jpgbytestring, np.uint8)
    try:
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except:
        return None


def generate_lineidx(filein, idxout):
    assert not os.path.isfile(idxout)
    with open(filein,'r') as fpin, open(idxout,'w') as fpout:
        fsize = os.fstat(fpin.fileno()).st_size
        def gen_rows():
            fpos = 0
            while fpos!=fsize:
                yield str(fpos)
                fpin.readline()
                fpos = fpin.tell()
        with tqdm(total=fsize) as t:
            for row in gen_rows():
                t.update(int(row) - t.n)
                fpout.write(row+"\n")


def load_from_yaml_file(file_name):
    with open(file_name, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)


def tsv_reader(tsv_file_name):
    with open(tsv_file_name, 'r') as fp:
        for _, line in enumerate(fp):
            yield [x.strip() for x in line.split('\t')]


def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        if not os.path.exists(path) and not os.path.islink(path):
            os.makedirs(path)


def tsv_writer_with_lineidx(values, tsv_file_name):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    idx = 0
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    tsv_lineidx_file_tmp = tsv_lineidx_file + '.tmp'
    with open(tsv_file_name_tmp, 'w') as fp, open(tsv_lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value
            v = '{0}\n'.format('\t'.join(map(str, value)))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)
    os.rename(tsv_file_name_tmp, tsv_file_name)
    os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)


def tsv_writer(values, tsv_file_name):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    with open(tsv_file_name_tmp, 'w') as fp:
        assert values is not None
        for value in values:
            assert value
            v = '{0}\n'.format('\t'.join(map(str, value)))
            fp.write(v)
    os.rename(tsv_file_name_tmp, tsv_file_name)
