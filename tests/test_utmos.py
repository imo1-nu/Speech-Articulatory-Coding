import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from sparc.evaluate_utmos import UTMOSPredictor,aggregate_utmos,collect_items,enrich_report,join_aggregates,score_items
from sparc.japanese_ppg import sha256_file


class UTMOSTest(unittest.TestCase):
    def test_predictor_keeps_native_gain_length_and_upstream_score(self):
        captured=[]
        class Model(torch.nn.Module):
            def forward(self,wave,sr):
                captured.append((wave.clone(),sr))
                return torch.tensor([5.1])
        predictor=UTMOSPredictor.__new__(UTMOSPredictor);predictor.model=Model();predictor.device='cpu'
        wave=np.linspace(-.1,.2,1001,dtype=np.float32)
        self.assertAlmostEqual(predictor.score(wave),5.1,places=5)
        np.testing.assert_array_equal(captured[0][0].numpy()[0],wave)
        self.assertEqual(captured[0][1],16000)
        with self.assertRaises(ValueError):predictor.score(np.zeros(399))
        with self.assertRaises(ValueError):predictor.score(np.full(400,np.nan))

    def test_equal_utterance_aggregation_and_existing_columns_preserved(self):
        rows=[{'condition':'x','utmos':1.,'samples_16k':400},
              {'condition':'x','utmos':5.,'samples_16k':40000}]
        values=aggregate_utmos(rows)
        self.assertEqual(values[0]['utmos_mean'],3)
        self.assertAlmostEqual(values[0]['utmos_std'],np.sqrt(8))
        old=[{'condition':'x','cer':'0.12','stoi':'0.8'}]
        merged=join_aggregates(old,values)
        self.assertEqual(merged[0]['cer'],'0.12');self.assertEqual(old[0],{'condition':'x','cer':'0.12','stoi':'0.8'})
        with self.assertRaises(ValueError):join_aggregates([{'condition':'missing'}],values)

    def test_single_observation_has_no_sample_standard_deviation(self):
        self.assertIsNone(aggregate_utmos([{'condition':'x','utmos':3.}])[0]['utmos_std'])

    def test_file_mapping_checksum_and_duplicate_waveform_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'utterance';folder.mkdir()
            sf.write(folder/'before.wav',np.linspace(-.2,.2,1600),16000,subtype='PCM_16')
            digest=sha256_file(folder/'before.wav')
            (root/'manifest.json').write_text(json.dumps({'complete':True,'inputs':[{'utterance_id':'u','directory':'utterance'}],
               'records':[{'utterance_id':'u','condition':'edited','audio':'utterance/before.wav'}]}))
            rows=[{'condition':name,'utterance_id':'u','audio_sha256':digest} for name in ('before','edited')]
            with (root/'audio_metrics.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            items,_=collect_items(root);calls=[];cache={}
            def predict(wave):calls.append(len(wave));return 3.25
            scored=score_items(root,items,predict,cache)
            self.assertEqual(calls,[1600]);self.assertEqual([r['utmos'] for r in scored],[3.25,3.25])
            self.assertFalse(scored[0]['cached']);self.assertTrue(scored[1]['cached'])
            score_items(root,items,None,cache);self.assertEqual(calls,[1600])
            sf.write(folder/'before.wav',np.zeros(1600),16000,subtype='PCM_16')
            with self.assertRaises(ValueError):collect_items(root)

    def test_report_injection_is_idempotent_and_preserves_existing_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'index.html'
            path.write_text('<html><p id="verification-summary">checked</p><h2>特徴量・計算時間</h2>'
                            '<h2>音声指標</h2><table><tr><th>条件</th><th>CER</th></tr>'
                            '<tr><td>x</td><td>10.00</td></tr></table></html>')
            values=aggregate_utmos([{'condition':'x','utmos':3.}])
            enrich_report(path,values,[{'condition':'x','cer':'.1'}]);first=path.read_text()
            enrich_report(path,values,[{'condition':'x','cer':'.1'}])
            self.assertEqual(path.read_text(),first)
            self.assertIn('<p id="verification-summary">checked</p>',first)
            self.assertEqual(first.count('id="utmos-evaluation"'),1)
            self.assertEqual(first.count('class="utmos-cell"'),2)
            self.assertIn('<td>10.00</td>',first)


if __name__=='__main__':unittest.main()
