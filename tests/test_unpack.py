import os
import unittest

import numpy as np
from daplis.functions.unpack import unpack_binary_data


class TestUnpackBin(unittest.TestCase):
    def test_valid_input(self):
        # Positive test case with valid inputs
        work_dir = r"{}".format(os.path.dirname(os.path.realpath(__file__)) + "/..")
        os.chdir(work_dir)
        file = r"tests/test_data/test_data_2212b.dat"
        daughterboard_number = "NL11"
        motherboard_number = "#33"
        timestamps = 300
        firmware_version = "2212b"
        include_offset = False

        data_all = unpack_binary_data(
            file,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            include_offset,
        )

        # Assert the shape of the output data
        self.assertEqual(data_all.shape, (64, 300 * 300 + 300, 2))
        # Assert the data type of the output data
        self.assertEqual(data_all.dtype, np.longlong)


if __name__ == "__main__":
    unittest.main()
