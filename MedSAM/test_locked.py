import unittest
from pathlib import Path
from train_engine_locked import RunConfig, check_config, patient_metric
from launch_locked import schedule, command

class LockedTests(unittest.TestCase):
    def test_matrix(self):
        self.assertEqual(len(schedule()),9)
        self.assertEqual(len(set(schedule())),9)
        self.assertEqual(schedule()[0],(5,1))
    def test_command(self):
        args=command(10,2,'disabled','cpu',Path('runs'))
        for s in ('--min-delta','0.001','--val-frequency','1','--train-limit','0'):
            self.assertIn(s,args)
    
    def test_launcher_argument_boundaries(self):
        """Check actual launcher arguments for every K/seed combination."""
        from launch_locked import command, schedule

        for k, seed in schedule():
            with self.subTest(k=k, seed=seed):
                args = command(k, seed, 'online', 'cuda:0', Path('runs'))
                self.assertIsInstance(args, list)
                self.assertNotIn('0--wandb-mode', args)
                self.assertEqual(args.count('--val-examples-per-patient'), 1)
                self.assertEqual(args.count('--wandb-mode'), 1)
                i = args.index('--val-examples-per-patient')
                j = args.index('--wandb-mode')
                self.assertEqual(args[i + 1], '0')
                self.assertEqual(args[j + 1], 'online')
                self.assertEqual(j, i + 2)
                self.assertEqual(args[args.index('--k') + 1], str(k))
                self.assertEqual(args[args.index('--seed') + 1], str(seed))
                self.assertEqual(
                    args[args.index('--experiment') + 1],
                    f'K{k}_seed{seed}',
                )

    def test_patient_weighting(self):
        rows=[{'patient_id':'p1','structure_type':'vertebral_body','dice':1.},
              {'patient_id':'p1','structure_type':'vertebral_body','dice':1.},
              {'patient_id':'p2','structure_type':'vertebral_body','dice':0.}]
        self.assertAlmostEqual(patient_metric(rows)['patient_mean_dice'],.5)
    def test_config(self):
        cfg=RunConfig(device='cpu',min_delta=.001)
        check_config(cfg)
        cfg.min_delta=-.1
        with self.assertRaises(ValueError):check_config(cfg)

if __name__=='__main__':unittest.main(verbosity=2)
