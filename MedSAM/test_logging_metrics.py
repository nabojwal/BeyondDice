import unittest
import torch
from train_engine import RunConfig, check_config, segmentation_counts, segmentation_scores, SafeWandb
from pathlib import Path
class LoggingMetricTests(unittest.TestCase):
    def test_perfect(self):
        z=torch.tensor([[[[10.,-10.],[-10.,10.]]]])
        y=torch.tensor([[[[1.,0.],[0.,1.]]]])
        r=segmentation_scores(segmentation_counts(z,y))
        for k,v in r.items():self.assertEqual(float(v[0]),1.,k)
    def test_incorrect(self):
        z=torch.tensor([[[[10.,10.],[-10.,-10.]]]])
        y=torch.tensor([[[[1.,0.],[1.,0.]]]])
        r=segmentation_scores(segmentation_counts(z,y))
        self.assertAlmostEqual(float(r['dice'][0]),.5)
        self.assertAlmostEqual(float(r['accuracy'][0]),.5)
        self.assertAlmostEqual(float(r['iou'][0]),1/3,places=6)
    def test_config(self):
        c=RunConfig(train_limit=0,val_patients_limit=0,val_examples_per_patient=0,wandb_mode='disabled')
        check_config(c)
        SafeWandb(c,Path('.'),{}).log({'epoch':1})
        with self.assertRaises(ValueError):check_config(RunConfig(wandb_mode='broken'))
if __name__=='__main__':unittest.main(verbosity=2)
