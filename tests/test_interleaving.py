import os
import shutil
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.dataset import (
    list_parquet_files,
    parquets_iter_batched,
    interleaved_parquets_iter_batched,
)


class TestDatasetInterleaving(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ds1_dir = os.path.join(self.test_dir, "karpathy")
        self.ds2_dir = os.path.join(self.test_dir, "altai")
        os.makedirs(self.ds1_dir, exist_ok=True)
        os.makedirs(self.ds2_dir, exist_ok=True)

        # Create dummy parquet files for karpathy (3 shards)
        for i in range(3):
            table = pa.Table.from_arrays(
                [pa.array([f"en_doc_{i}_{j}" for j in range(10)])], names=["text"]
            )
            pq.write_table(table, os.path.join(self.ds1_dir, f"shard_{i:05d}.parquet"))

        # Create dummy parquet files for altai (2 shards)
        for i in range(2):
            table = pa.Table.from_arrays(
                [pa.array([f"tr_doc_{i}_{j}" for j in range(10)])], names=["text"]
            )
            pq.write_table(table, os.path.join(self.ds2_dir, f"shard_{i:05d}.parquet"))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_list_parquet_files(self):
        result = list_parquet_files(data_dir=self.test_dir)
        self.assertIsInstance(result, dict)
        self.assertIn("karpathy", result)
        self.assertIn("altai", result)
        self.assertEqual(len(result["karpathy"]), 3)
        self.assertEqual(len(result["altai"]), 2)

    def test_parquets_iter_batched_splits(self):
        # Check dataset 1 train split (first 2 shards)
        train_batches = list(
            parquets_iter_batched(split="train", dataset_name="karpathy", data_dir=self.test_dir)
        )
        self.assertEqual(len(train_batches), 2)
        self.assertTrue(all(d.startswith("en_doc_") for batch in train_batches for d in batch))

        # Check dataset 1 val split (last shard)
        val_batches = list(
            parquets_iter_batched(split="val", dataset_name="karpathy", data_dir=self.test_dir)
        )
        self.assertEqual(len(val_batches), 1)
        self.assertTrue(all(d.startswith("en_doc_2_") for d in val_batches[0]))

    def test_interleaved_parquets_iter_batched(self):
        # Stream 50 batches with 70/30 weights
        gen = interleaved_parquets_iter_batched(
            split="train",
            datasets=["karpathy", "altai"],
            weights=[0.7, 0.3],
            data_dir=self.test_dir,
            seed=1234,
        )
        batches = [next(gen) for _ in range(50)]
        self.assertEqual(len(batches), 50)
        
        # Check that both English and Turkish documents are generated
        en_count = sum(1 for b in batches if b[0].startswith("en_doc_"))
        tr_count = sum(1 for b in batches if b[0].startswith("tr_doc_"))
        self.assertGreater(en_count, 0)
        self.assertGreater(tr_count, 0)
        self.assertEqual(en_count + tr_count, 50)


if __name__ == "__main__":
    unittest.main()
