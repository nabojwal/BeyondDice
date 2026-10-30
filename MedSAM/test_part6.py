import unittest
from train_engine_part6 import RunConfig, check_config, patient_metric
from wandb_logger import WandbLogger


class Part6UnitTests(unittest.TestCase):
    def test_unlimited_full_data_config(self):
        c = RunConfig(train_limit=0, val_patients_limit=0, val_examples_per_patient=0)
        check_config(c)
        self.assertEqual(c.train_limit, 0)

    def test_invalid_wandb_mode(self):
        with self.assertRaises(ValueError):
            check_config(RunConfig(wandb_mode='invalid'))

    def test_disabled_logger_no_dependency(self):
        logger = WandbLogger(mode='disabled', project='p', group='g', name='n', config={}, tags=[])
        logger.log({'train_total':0.1},step=1)
        logger.finish()

    def test_patient_equal_weighting(self):
        rows=[{'patient_id':'a','structure_type':'disc','dice':0.0} for _ in range(4)]
        rows.append({'patient_id':'b','structure_type':'disc','dice':1.0})
        self.assertAlmostEqual(patient_metric(rows)['patient_mean_dice'],0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
