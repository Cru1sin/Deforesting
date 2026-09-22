from pathlib import Path
from zipfile import ZipFile

import pytest

from dataset_tools import dts_downloader as dts
from dataset_tools.organize_raw_dates import organize_date


def test_extract_flatten_rename_verify_before_delete(tmp_path):
    date = tmp_path / '0824'
    date.mkdir()
    for name, member in [('0824S.zip', '0824S/参数1.xls'),
                         ('192.168.1.1_1.zip', '192.168.1.1_1/image.jpg')]:
        with ZipFile(date / name, 'w') as z:
            z.writestr(member, b'original')
    organize_date(date)
    assert len(list(date.glob('*.zip'))) == 2
    organize_date(date, True)
    assert (date / '参数1.xls').read_bytes() == b'original'
    assert (date / 'top/image.jpg').read_bytes() == b'original'
    assert not list(date.glob('*.zip'))
    with ZipFile(date / '0824S.zip', 'w') as z:
        z.writestr('参数1.xls', b'conflict')
    with pytest.raises(FileExistsError):
        organize_date(date, True)
    assert (date / '0824S.zip').exists()
    assert (date / '参数1.xls').read_bytes() == b'original'


def test_download_dates_follow_contents_not_six_file_groups(monkeypatch):
    monkeypatch.setattr(dts, 'camera_archive_date', lambda f: f['date'])
    files = [{'filename': '0824S.zip'}, {'filename': 'camera.zip', 'date': '0825'},
             {'filename': 'camera.zip', 'date': '0824'}]
    jobs = dts.plan_downloads(files, Path('/tmp/data'), start_no=1, end_no=None)
    assert [j['destination'].parent.name for j in jobs] == ['0824', '0825', '0824']


def test_existing_flattened_copy_is_verified_before_removing_wrapper(tmp_path):
    date = tmp_path / '0831'
    (date / '0831S' / '192.168.1.15_1').mkdir(parents=True)
    (date / '0831S' / '192.168.1.15_1' / 'a.jpg').write_bytes(b'original')
    (date / '192.168.1.15_1').mkdir()
    (date / '192.168.1.15_1' / 'a.jpg').write_bytes(b'original')
    organize_date(date, True)
    assert (date / 'front/a.jpg').read_bytes() == b'original'
    assert not (date / '0831S').exists()
