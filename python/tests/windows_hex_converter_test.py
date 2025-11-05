import unittest

from python import windows_hex_converter as converter


class WindowsHexConverterTest(unittest.TestCase):
    def test_sample_hex_conversion(self) -> None:
        binary, fields = converter.convert_hex_to_bitfields(
            "05F821E300020000000C000000000000"
        )

        self.assertEqual(len(binary), 128)
        self.assertEqual(int(binary, 2), int("05F821E300020000000C000000000000", 16))

        labels = [field.label for field in fields]
        self.assertEqual(
            labels,
            [
                "127-110",
                "109-105",
                "104-96",
                "95-92",
                "91-80",
                "79-64",
                "63-48",
                "47-32",
                "31-16",
                "15-0",
            ],
        )

        self.assertEqual(fields[0].binary, "000001011111100000")
        self.assertEqual(fields[0].value, 6112)
        self.assertEqual(fields[6].binary, "0000000000001100")
        self.assertEqual(fields[6].value, 12)


if __name__ == "__main__":
    unittest.main()
