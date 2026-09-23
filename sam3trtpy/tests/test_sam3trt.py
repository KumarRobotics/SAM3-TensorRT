import unittest
from pathlib import Path

import cv2
import numpy as np

from sam3trt import CONFIG_DIR, Detections, FeatureMap, Sam3Model, Sam3Processor, visualize
from sam3trt.gpu import GpuArray, device, grid_2d, run_kernel
from sam3trt.tokenizer import ClipTokenizer

TEST_IMAGE = Path(__file__).parent / "test.png"
MODELS_DIR = Path.home() / "models"
BLOCK = (32, 16, 1)


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self._tokenizer = ClipTokenizer(CONFIG_DIR / "merges.txt", CONFIG_DIR / "vocab.json")

    def test_known_ids(self):
        self.assertEqual(self._tokenizer.tokenize("person"), [49406, 2533, 49407])
        self.assertEqual(self._tokenizer.tokenize("Red  Car"), [49406, 736, 1615, 49407])


class TestKernels(unittest.TestCase):
    def setUp(self):
        self._stream = device().create_stream()
        self._rng = np.random.default_rng(0)

    def test_resize_normalize(self):
        image = self._rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        params = Sam3Processor.Params(image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5))
        processor = Sam3Processor(CONFIG_DIR / "merges.txt", CONFIG_DIR / "vocab.json", params)
        result = processor.preprocess_image(image).image.download(self._stream)[0]

        size = params.image_size
        resized = cv2.resize(image.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
        expected = ((resized[..., ::-1] / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_upsample_masks(self):
        masks = self._rng.standard_normal((200, 184, 184)).astype(np.float16)
        indices = np.array([3, 17, 150], np.int32)
        orig_h, orig_w = 375, 500

        d_masks = GpuArray(masks.shape, np.float16)
        d_indices = GpuArray(indices.shape, np.int32)
        d_out = GpuArray((indices.size, orig_h, orig_w), np.uint8)
        d_boxes = GpuArray((indices.size, 4), np.int32)
        d_masks.upload(masks, self._stream)
        d_indices.upload(indices, self._stream)
        d_boxes.upload(np.tile(np.array([orig_w, orig_h, -1, -1], np.int32), (indices.size, 1)), self._stream)

        run_kernel(
            "upsampleMasks",
            grid_2d(orig_w, orig_h, BLOCK, indices.size),
            BLOCK,
            self._stream,
            d_masks.ptr,
            d_indices.ptr,
            d_out.ptr,
            d_boxes.ptr,
            *(np.int32(v) for v in (184, 184, orig_h, orig_w)),
        )
        out, boxes = d_out.download(self._stream), d_boxes.download(self._stream)

        for mask, box, index in zip(out, boxes, indices):
            reference = cv2.resize(masks[index].astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            self.assertLess(np.mean(mask != np.where(reference > 0, 255, 0)), 1e-3)
            x, y, w, h = cv2.boundingRect(mask)
            self.assertEqual(list(box), [x, y, x + w - 1, y + h - 1])


class TestResult(unittest.TestCase):
    def test_feature_map(self):
        features = FeatureMap(np.arange(256 * 46 * 46, dtype=np.float32))
        self.assertEqual(features.patch(1, 2)[1], 46 * 46 + 46 + 2)
        with self.assertRaises(IndexError):
            features.patch(46, 0)

    def test_detections(self):
        detections = Detections([])
        self.assertTrue(detections.empty())
        self.assertEqual(len(detections), 0)


class TestSam3Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._model = Sam3Model(
            MODELS_DIR,
            CONFIG_DIR / "merges.txt",
            CONFIG_DIR / "vocab.json",
            CONFIG_DIR / "params.json",
        )
        cls._image = cv2.imread(str(TEST_IMAGE))

    def test_single_prompt(self):
        result = self._model.infer(self._image, "car")
        self.assertGreater(len(result.detections), 0)
        for detection in result.detections:
            self.assertEqual(detection.mask.shape, self._image.shape[:2])
            self.assertEqual(detection.bbox, list(cv2.boundingRect(detection.mask)))
            self.assertEqual(detection.class_label, "car")
            self.assertGreaterEqual(detection.confidence, 0.3)
        self.assertEqual(result.features.patch(23, 12).shape, (256,))

    def test_multiple_prompts(self):
        result = self._model.infer(self._image, ["person", "car"])
        self.assertEqual({d.class_id for d in result.detections}, {0, 1})
        self.assertEqual(visualize(self._image, result).shape, self._image.shape)

    def test_empty_prompts(self):
        with self.assertRaises(ValueError):
            self._model.infer(self._image, [])


if __name__ == "__main__":
    unittest.main()
