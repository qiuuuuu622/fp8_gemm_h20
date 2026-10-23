import copy
import json
import math
from pathlib import Path
import unittest
from fp8_gemm_h20.config import get_profile, select, supports, validate_config

ROOT = Path(__file__).resolve().parents[1]

class Profiles(unittest.TestCase):
    def test_35b_preserves_historical_dispatch(self):
        original = json.loads((ROOT / 'runs/dispatch.json').read_text())
        for nk, rows in original.items():
            n, k = map(int, nk.split(','))
            for m in sorted(set(map(int, rows)) | {1, 7, 37, 200, 777, 4097, 16384}):
                anchor = min(rows, key=lambda a: abs(math.log(int(a)) - math.log(m)))
                expected = rows[anchor]
                actual, _ = select('35b_tp1', m, n, k)
                self.assertEqual(actual['impl'], {'c3':'triton','c11':'gluon'}[expected['impl']])
                self.assertEqual(actual['config'], expected['cfg'])

    def test_122b_routing_matches_recorded_260_cases(self):
        import csv
        with (ROOT / 'experiments/122b_tp4/v2/comparison.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 260)
        for row in rows:
            entry, info = select('122b_tp4', *[int(row[x]) for x in ('M','N','K')])
            self.assertEqual(entry['impl'], row['impl'], row)
            self.assertEqual(info['method'], row['selection'], row)

    def test_122b_full_config_migration(self):
        data=get_profile('122b_tp4')
        original=json.loads((ROOT/'experiments/122b_tp4/v2/dispatch_table.json').read_text())
        fallback=json.loads((ROOT/'experiments/122b_tp4/v1/optimized.json').read_text())
        expected={nk:{m:dict(**row['entry'],anchor_us=row['us']) for m,row in table.items()} for nk,table in original.items()}
        self.assertEqual(data['entries'],expected)
        self.assertEqual(data['fallback_entries'],fallback)

    def test_shortk_is_exact_only(self):
        for m in (510, 511, 513):
            entry, _ = select('122b_tp4', m, 3072, 256)
            self.assertNotEqual(entry['impl'], 'shortk')
        self.assertEqual(select('122b_tp4', 512, 3072, 256)[0]['impl'], 'shortk')

    def test_all_offgrid_choices_obey_policy(self):
        data = get_profile('122b_tp4')
        for nk, entries in data['entries'].items():
            n, k = map(int, nk.split(','))
            for m in range(1, 16385):
                # CPU-only exhaustive coverage of the bounded lookup.
                entry, info = select('122b_tp4', m, n, k)
                if info['method'] == 'equal_tile_count':
                    from fp8_gemm_h20.config import token_tile
                    bt = token_tile(entry)
                    self.assertLessEqual(abs(m-info['anchor']), 8)
                    self.assertEqual(math.ceil(m/bt), math.ceil(info['anchor']/bt))
                    self.assertNotEqual(entry['impl'], 'shortk')
                if info['method'] == 'baseline_fallback':
                    baseline = data['fallback_entries'][nk]
                    a = min(baseline, key=lambda a: abs(math.log(int(a)/m)))
                    self.assertEqual(entry, baseline[a])

    def test_bounds_unknown_profile_and_mutability(self):
        self.assertFalse(supports('122b_tp4', 0, 3072, 256))
        for args in [('122b_tp4',16385,3072,256),('35b_tp1',37,3072,256),('bad',1,512,3072)]:
            with self.assertRaises(ValueError): select(*args)
        a,_=select('122b_tp4',512,3072,256)
        a['config']['BM']=999
        self.assertNotEqual(select('122b_tp4',512,3072,256)[0]['config']['BM'],999)

    def test_reject_malformed_config(self):
        data=get_profile('122b_tp4')
        for change in ('shape','kernel','tile','latency'):
            d=copy.deepcopy(data); e=next(iter(next(iter(d['entries'].values())).values()))
            if change=='shape': d['shapes']=[]
            if change=='kernel': e['impl']='unknown'
            if change=='tile': e['config']['BLOCK_T']=3
            if change=='latency': e['anchor_us']=float('nan')
            with self.assertRaises(ValueError): validate_config(d)

if __name__ == '__main__': unittest.main()
