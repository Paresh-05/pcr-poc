from app.redact import redact


def test_redacts_private_details():
    text = "call 9876543210 or +91 9123456780, VIN MZBLG1234ABCD5678, plate MH12AB1234, a@b.com http://x.io/y"
    out = redact(text)
    for token in ("9876543210", "9123456780", "MZBLG1234ABCD5678", "MH12AB1234", "a@b.com", "http://x.io"):
        assert token not in out


def test_keeps_part_numbers_and_dtcs():
    text = "DTC P200200 part 000000000011640409 K68566622AA connector X104"
    assert redact(text) == text
