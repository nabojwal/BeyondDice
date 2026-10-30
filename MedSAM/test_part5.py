"""Fast unit tests; NOT a substitute for the real SPIDER smoke integration."""
import tempfile, unittest
from pathlib import Path
import torch
from train_engine import (RunConfig,check_config,patient_metric,dice_per_instance,
                          rng_capture,rng_restore,seed_all,atomic_save)

class Part5Tests(unittest.TestCase):
    def test_patient_equal_weight(self):
        rows=[{'patient_id':'a','structure_type':'disc','dice':1.0},
              {'patient_id':'a','structure_type':'disc','dice':1.0},
              {'patient_id':'b','structure_type':'disc','dice':0.0}]
        m=patient_metric(rows)
        self.assertEqual(m['patient_mean_dice'],0.5)
        self.assertEqual(m['patient_count'],2)
    def test_empty_and_overlap(self):
        z=torch.full((2,1,4,4),-5.)
        gt=torch.zeros_like(z)
        self.assertEqual(dice_per_instance(z,gt),[1.,1.])
        z[0]=5;gt[0]=1
        self.assertEqual(dice_per_instance(z,gt),[1.,1.])
    def test_config_failures(self):
        with self.assertRaises(ValueError):check_config(RunConfig(accum_steps=0))
        with self.assertRaises(ValueError):check_config(RunConfig(device='cpu',amp=True))
    def test_rng_restore(self):
        seed_all(51,False);state=rng_capture();a=torch.rand(3);rng_restore(state)
        self.assertTrue(torch.equal(a,torch.rand(3)))
    def test_atomic_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.pt';atomic_save({'a':torch.ones(1)},p)
            self.assertEqual(torch.load(p,weights_only=True)['a'].item(),1.)

if __name__=='__main__':unittest.main(verbosity=2)
